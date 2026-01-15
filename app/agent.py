"""
Legal Agent - Gemini 3 Flash via OpenRouter.

Thinks like a lawyer: prioritizes laws, uses judgments for interpretation,
cites sources, and reasons systematically.
"""
import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import asdict

import httpx

from app.config import settings
from app import tools


# =============================================================================
# SYSTEM PROMPT - LAWYER MINDSET
# =============================================================================

def get_system_prompt() -> str:
    """Generate system prompt with current timestamp."""
    current_date = datetime.now().strftime("%d. %m. %Y")
    current_year = datetime.now().year
    
    return f"""Jsi zkušený český právník s hlubokými znalostmi českého práva.

## AKTUÁLNÍ DATUM: {current_date}
## AKTUÁLNÍ ROK: {current_year}

Všechny právní informace musí být platné k tomuto datu. Neodkazuj na informace z minulých let (2024, 2025) pokud nejsou stále aktuální.

## PRÁVNÍ METODOLOGIE

1. **HIERARCHIE ZDROJŮ** (vždy dodržuj):
   - Ústavní pořádek (Ústava, Listina základních práv)
   - Zákony a zákoníky (občanský zákoník, trestní zákoník, atd.)
   - Podzákonné předpisy (vyhlášky, nařízení)
   - Judikatura (interpretační vodítko, ne primární zdroj)


2. **POSTUP ANALÝZY**:
   - Nejdřív identifikuj právní otázku
   - Hledej relevantní ustanovení v zákonech
   - Teprve pak hledej judikaturu pro výklad
   - Odpověz PŘÍMO na otázku, a svá tvrzení podlož citacemi

3. **CITACE** (povinné):
   - Zákony: "§ 123 odst. 1 zákona č. 89/2012 Sb., občanský zákoník"
   - Judikatura: "Nález ÚS sp. zn. III. ÚS 123/20"
   
4. **DŮLEŽITÉ**:
   - Odpovídej jasně a srozumitelně (jako klientovi, ne jako soudu)
   - NIKDY nevymýšlej ustanovení - cituj jen to, co máš v kontextu
   - Rozlišuj platné a zrušené předpisy
   - Odpovídej v češtině

## DOSTUPNÉ NÁSTROJE

- `search_laws` - Hledání v zákonech (PRIMÁRNÍ zdroj)
- `search_judgments` - Hledání v judikatuře (SEKUNDÁRNÍ)
- `get_full_section` - Celý text paragrafu
- `get_full_judgment` - Celý text rozhodnutí
- `web_search` - Webové vyhledávání (Perplexity Sonar pro aktuální informace)

Používej nástroje iterativně dokud nemáš dostatek informací pro kvalitní odpověď."""


# Tool definitions for Gemini
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_laws",
            "description": "Vyhledá relevantní ustanovení v českých zákonech. PRIMÁRNÍ zdroj pro právní otázky. Použij vždy jako první krok.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Vyhledávací dotaz (např. 'nájem bytu výpověď')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Počet výsledků (default 10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_judgments",
            "description": "Vyhledá soudní rozhodnutí. SEKUNDÁRNÍ zdroj - použij pro výklad zákonů a judikaturu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Vyhledávací dotaz"
                    },
                    "court": {
                        "type": "string",
                        "description": "Filtr soudu: 'ConCo' (ÚS), 'SupCo' (NS), 'SupAdmCo' (NSS)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Počet výsledků (default 10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_section",
            "description": "Získá kompletní text konkrétního paragrafu zákona.",
            "parameters": {
                "type": "object",
                "properties": {
                    "law_citation": {
                        "type": "string",
                        "description": "Citace zákona (např. '89/2012 Sb.')"
                    },
                    "section_citation": {
                        "type": "string",
                        "description": "Citace paragrafu (např. '§ 2235')"
                    }
                },
                "required": ["law_citation", "section_citation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_judgment",
            "description": "Získá kompletní text soudního rozhodnutí.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_number": {
                        "type": "string",
                        "description": "Spisová značka (např. 'III. ÚS 123/20')"
                    }
                },
                "required": ["case_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Webové vyhledávání pomocí Perplexity Sonar. Použij pro aktuální informace, novinky, nebo když v právní databázi nic nenajdeš.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Vyhledávací dotaz"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# =============================================================================
# AGENT
# =============================================================================

class LegalAgent:
    """
    Agentic legal research using Gemini 3 Flash via OpenRouter.
    """
    
    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = settings.AGENT_MODEL  # gemini-3-flash-preview
    
    async def _call_api(
        self, 
        messages: List[Dict],
        use_tools: bool = True
    ) -> Dict:
        """Call OpenRouter API."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://lexio.cz",
                        "X-Title": "Lexio Legal Agent"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": TOOLS if use_tools else None,
                        "temperature": 0.3,
                        "max_tokens": 4000
                    }
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger = logging.getLogger("agent")
            logger.error(f"API Call failed: {e}")
            raise
    
    async def _execute_tool(self, name: str, args: Dict) -> str:
        """Execute a tool and return result as JSON string."""
        try:
            if name == "search_laws":
                results = await tools.search_laws(
                    args["query"],
                    limit=args.get("limit", 10)
                )
                return json.dumps([asdict(r) for r in results], ensure_ascii=False)
            
            elif name == "search_judgments":
                results = await tools.search_judgments(
                    args["query"],
                    court=args.get("court"),
                    limit=args.get("limit", 10)
                )
                return json.dumps([asdict(r) for r in results], ensure_ascii=False)
            
            elif name == "get_full_section":
                result = await tools.get_full_section(
                    args["law_citation"],
                    args["section_citation"]
                )
                return json.dumps(result or {"error": "Not found"}, ensure_ascii=False)
            
            elif name == "get_full_judgment":
                result = await tools.get_full_judgment(args["case_number"])
                return json.dumps(result or {"error": "Not found"}, ensure_ascii=False)
            
            elif name == "web_search":
                result = await tools.web_search(args["query"])
                return json.dumps(result or {"error": "No results"}, ensure_ascii=False)
            
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})
        
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def ask(
        self, 
        question: str,
        max_iterations: int = 10
    ) -> AsyncIterator[Dict]:
        """
        Ask a legal question with iterative tool use.
        
        Yields events:
        - {"event": "thinking", "content": "..."}
        - {"event": "tool_call", "tool": "...", "args": {...}}
        - {"event": "tool_result", "tool": "...", "result": "..."}
        - {"event": "answer", "content": "..."}
        - {"event": "done"}
        """
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": question}
        ]
        
        yield {"event": "thinking", "content": "Analyzuji právní otázku..."}
        
        answer_generated = False
        
        for iteration in range(max_iterations):
            response = await self._call_api(messages, use_tools=True)
            
            choice = response["choices"][0]
            message = choice["message"]
            
            # Check for tool calls
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    func = tool_call["function"]
                    name = func["name"]
                    try:
                        args = json.loads(func["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    
                    yield {"event": "tool_call", "tool": name, "args": args}
                    
                    # Execute tool
                    result = await self._execute_tool(name, args)
                    
                    yield {"event": "tool_result", "tool": name, "result": result[:500] + "..." if len(result) > 500 else result}
                    
                    # Add to messages
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })
            
            # Check for answer content
            elif message.get("content"):
                yield {"event": "answer", "content": message["content"]}
                
                # If finished by length, continue generating
                if choice.get("finish_reason") == "length":
                     messages.append({"role": "assistant", "content": message["content"]})
                     messages.append({"role": "user", "content": "Pokračuj v odpovědi přesně tam, kde jsi skončil (dokonči větu)."})
                     continue
                
                answer_generated = True
                break
            
            # Check finish reason stop (if no content or tool calls matched above)
            if choice.get("finish_reason") == "stop":
                 break
        
        # If no answer was generated, force one
        if not answer_generated:
            yield {"event": "thinking", "content": "Generuji finální odpověď..."}
            messages.append({
                "role": "user",
                "content": "Nyní prosím shrň svou odpověď. Použij informace z předchozích vyhledávání a odpověz přímo na původní otázku."
            })
            
            response = await self._call_api(messages, use_tools=False)
            final_answer = response["choices"][0]["message"].get("content", "")
            if final_answer:
                yield {"event": "answer", "content": final_answer}
        
        yield {"event": "done"}
    
    async def quick_search(self, query: str, source: str = "law") -> List[Dict]:
        """
        Quick search without AI - just returns results.
        """
        if source == "law":
            results = await tools.search_laws(query, limit=20)
            return [asdict(r) for r in results]
        elif source == "case":
            results = await tools.search_judgments(query, limit=20)
            return [asdict(r) for r in results]
        else:
            return []


# Singleton
agent = LegalAgent()



# =============================================================================
# CLI
# =============================================================================

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-cli")

async def run_question(question: str):
    """Run a single question and print the answer."""
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")
    
    async for event in agent.ask(question):
        if event["event"] == "thinking":
            # Just log thinking efficiently, don't spam print
            print(f"Thinking: {event['content']}")
        elif event["event"] == "tool_call":
            print(f"Tool Call: {event['tool']} ({str(event['args'])[:60]}...)")
        elif event["event"] == "tool_result":
            print(f"   -> Result: {event['result'][:100]}...")
        elif event["event"] == "answer":
            print(f"\n{'='*60}")
            print("ANSWER:")
            print("="*60)
            print(event["content"])
            print("="*60)
        elif event["event"] == "done":
            print("\nDone.")
    
    await tools.close_pool()


if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Lexio Legal Agent")
    parser.add_argument("-q", "--question", type=str, help="Ask a single question")
    args = parser.parse_args()
    
    if args.question:
        asyncio.run(run_question(args.question))
    else:
        print("Usage: python -m app.agent -q 'Legal question needing analysis'")

