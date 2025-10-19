from __future__ import annotations
import os
from dotenv import load_dotenv

# load .env in project root (if you used one)
load_dotenv()

# quick runtime check (masks key)
_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY present:", bool(_key), "masked:", (_key[:8] + "...") if _key else None)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Protocol, TYPE_CHECKING

# Try multiple import locations for Chat LLM (works across langchain versions)
ChatLLMClass = None
try:
    from langchain.chat_models import ChatOpenAI
    ChatLLMClass = ChatOpenAI
except Exception:
    try:
        from langchain.llms import OpenAI as OpenAI_LLMS
        ChatLLMClass = OpenAI_LLMS
    except Exception:
        ChatLLMClass = None

# Try multiple import locations for ConversationBufferMemory
ConversationBufferMemory = None
try:
    from langchain.memory.buffer import ConversationBufferMemory as _ConvBuf
    ConversationBufferMemory = _ConvBuf
except Exception:
    try:
        from langchain.memory import ConversationBufferMemory as _ConvBuf2
        ConversationBufferMemory = _ConvBuf2
    except Exception:
        ConversationBufferMemory = None

# Try to import ConversationChain; provide a lightweight fallback if missing
ConversationChain = None
try:
    from langchain.chains import ConversationChain as _ConvChain
    ConversationChain = _ConvChain
except Exception:
    class SimpleConversationBufferMemory:
        def __init__(self):
            self.buffer = []
        def add_user_message(self, text: str):
            self.buffer.append(f"User: {text}")
        def add_ai_message(self, text: str):
            self.buffer.append(f"AI: {text}")
        def load_memory_variables(self, _=None):
            return {"history": "\n".join(self.buffer)}

    class SimpleConversationChain:
        def __init__(self, llm, memory=None, verbose=False):
            self.llm = llm
            self.memory = memory
            self.verbose = verbose

        def _call_llm(self, prompt: str) -> str:
            try:
                res = self.llm(prompt)
                if isinstance(res, str):
                    return res
                if hasattr(res, "generations"):
                    gens = res.generations
                    if gens and gens[0] and hasattr(gens[0][0], "text"):
                        return gens[0][0].text
                if isinstance(res, dict) and "text" in res:
                    return res["text"]
                return str(res)
            except TypeError:
                try:
                    return self.llm.predict(prompt)
                except Exception:
                    pass
            except Exception:
                pass

            try:
                gen = self.llm.generate([prompt])
                if hasattr(gen, "generations"):
                    return gen.generations[0][0].text
                return str(gen)
            except Exception as e:
                raise RuntimeError(f"LLM call failed: {e}")

        def predict(self, input: str) -> str:
            history = ""
            if self.memory:
                if hasattr(self.memory, "buffer"):
                    if isinstance(self.memory.buffer, list):
                        history = "\n".join(self.memory.buffer) + ("\n" if self.memory.buffer else "")
                    elif isinstance(self.memory.buffer, str):
                        history = self.memory.buffer + ("\n" if self.memory.buffer else "")
                elif hasattr(self.memory, "load_memory_variables"):
                    vars = self.memory.load_memory_variables()
                    history = vars.get("history", "") + ("\n" if vars.get("history") else "")

            prompt = f"{history}User: {input}\nAI:"
            if self.verbose:
                print("PROMPT:", prompt)

            result = self._call_llm(prompt)

            if self.memory:
                if hasattr(self.memory, "add_user_message"):
                    self.memory.add_user_message(input)
                elif hasattr(self.memory, "buffer"):
                    self.memory.buffer.append(f"User: {input}")

                if hasattr(self.memory, "add_ai_message"):
                    self.memory.add_ai_message(result)
                elif hasattr(self.memory, "buffer"):
                    self.memory.buffer.append(f"AI: {result}")

            return result

    ConversationChain = SimpleConversationChain
    if ConversationBufferMemory is None:
        ConversationBufferMemory = SimpleConversationBufferMemory

# If no LangChain LLM class found, provide a simple OpenAI-backed LLM wrapper
if ChatLLMClass is None:
    try:
        import openai
    except Exception:
        raise RuntimeError(
            "No LangChain LLM class found and openai package is not installed. "
            "Install openai (pip install openai) or install a compatible langchain."
        )

    class _SimpleGen:
        def __init__(self, text):
            self.text = text

    class _SimpleLLMResult:
        def __init__(self, text):
            self.generations = [[_SimpleGen(text)]]

    class SimpleOpenAIChatLLM:
        def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7):
            self.model_name = model_name
            self.temperature = temperature
            # openai reads OPENAI_API_KEY from env; also set explicit api_key
            self._api_key = os.getenv("OPENAI_API_KEY")
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is not set")
            try:
                import openai
                openai.api_key = self._api_key
            except Exception:
                pass

        def _extract_text(self, resp):
            # try several common response shapes
            try:
                return resp.choices[0].message.content.strip()
            except Exception:
                pass
            try:
                return resp["choices"][0]["message"]["content"].strip()
            except Exception:
                pass
            try:
                # LLMResult-like
                if hasattr(resp, "generations"):
                    return resp.generations[0][0].text
            except Exception:
                pass
            return str(resp)

        def _call_chat(self, prompt: str) -> str:
            try:
                # Try new openai>=1.0.0 client
                try:
                    from openai import OpenAI as OpenAIClient
                    client = OpenAIClient()
                    resp = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                    )
                    return self._extract_text(resp)
                except Exception:
                    # fallback to legacy openai.ChatCompletion if present
                    import openai as _openai
                    if hasattr(_openai, "ChatCompletion"):
                        resp = _openai.ChatCompletion.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                        )
                        return self._extract_text(resp)
                    raise
            except Exception as e:
                raise RuntimeError(f"LLM call failed: {e}")

        def __call__(self, prompt: str):
            return self._call_chat(prompt)

        def predict(self, prompt: str) -> str:
            return self._call_chat(prompt)

        def generate(self, prompts):
            text = self._call_chat(prompts[0] if isinstance(prompts, (list, tuple)) else prompts)
            return _SimpleLLMResult(text)

    ChatLLMClass = SimpleOpenAIChatLLM

app = FastAPI(
    title="LangChain ChatBot API",
    description="A conversational AI chatbot using LangChain and OpenAI",
    version="1.0.0"
)

if TYPE_CHECKING:
    from langchain.chains import ConversationChain  # type: ignore

class ConversationChainLike(Protocol):
    def predict(self, input: str) -> str: ...

conversation_chain: Optional[ConversationChainLike] = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class OpenAIChatbot:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        os.environ["OPENAI_API_KEY"] = api_key

        try:
            self.llm = ChatLLMClass(model_name="gpt-3.5-turbo", temperature=0.7)
        except TypeError:
            self.llm = ChatLLMClass(model_name="gpt-3.5-turbo")

    def create_conversation(self) -> ConversationChainLike:
        memory = ConversationBufferMemory() if ConversationBufferMemory is not None else None
        return ConversationChain(llm=self.llm, memory=memory, verbose=False)

chatbot = OpenAIChatbot()

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(chat_request: ChatRequest):
    global conversation_chain
    try:
        if conversation_chain is None:
            conversation_chain = chatbot.create_conversation()
        response = conversation_chain.predict(input=chat_request.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
