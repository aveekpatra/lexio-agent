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

3. **CITACE A ODKAZY** (POVINNÉ - STRIKTNĚ DODRŽUJ):
   - **KAŽDÁ** citace zákona MUSÍ mít odkaz, pokud máš `source_url` z nástroje.
   - Formát: `[§ 123 zákona č. 89/2012 Sb.](source_url)` - použij přesný URL z výsledku nástroje.
   - Příklad správné citace: `[§ 2235 občanského zákoníku](https://www.e-sbirka.cz/sb/2012/89#par_2235)`
   - BEZ odkazu NIKDY necituj zákon, pokud source_url existuje!
   - Judikatura: "Nález ÚS sp. zn. III. ÚS 123/20" (odkaz pokud dostupný)
   
4. **DŮLEŽITÉ**:
   - Odpovídej jasně a srozumitelně (jako klientovi, ne jako soudu)
   - NIKDY nevymýšlej ustanovení - cituj jen to, co máš v kontextu
   - Rozlišuj platné a zrušené předpisy
   - Odpovídej v češtině

## DOSTUPNÉ NÁSTROJE

### PRIMÁRNÍ (databáze - bez limitu):
- `search_laws` - **VŽDY POUŽIJ PRVNÍ** - rychlé vyhledávání v naší databázi
- `get_full_section` - Celý text paragrafu z databáze

### E-SBÍRKA API (pouze jako ZÁLOHA - limit 25 req/sec):
- `search_esbirka_api` - Přímé vyhledávání v E-Sbírka API. **POUZE když search_laws NENAJDE!**
- `get_law_changes` - Nedávné změny zákonů (použij pro ověření novelizací)
- `get_historical_section` - Historická verze k určitému datu (použij jen když potřebuješ starší verzi)

### OSTATNÍ:
- `search_judgments` - Judikatura (sekundární zdroj)
- `get_full_judgment` - Celý text rozhodnutí
- `web_search` - **POUZE** pro neprávní informace (čísla, statistiky)

## PRIORITA NÁSTROJŮ PRO ZÁKONY

⚠️ **DŮLEŽITÉ**: E-Sbírka API nástroje mají limit 25 req/sec. VŽDY preferuj databázi!

1. `search_laws` → VŽDY začni zde (databáze)
2. `get_full_section` → pro přesné znění (databáze)
3. Zkus jiný dotaz v `search_laws` (jiná slova, synonyma)
4. `search_esbirka_api` → POUZE pokud databáze opravdu nemá data
5. `get_historical_section` → jen pro historickou analýzu
6. `web_search` → POUZE pro neprávní doplňující info

## DŮKLADNOST ANALÝZY

- Používej nástroje iterativně dokud nemáš dostatek informací pro kvalitní odpověď.
- **VNOŘENÉ CITACE**: Pokud zákon odkazuje na jiný zákon (např. § 420 odkazuje na § 2910), vyhledej i ten odkazovaný zákon!
- Neboj se provést více vyhledávání pro úplný obraz.
- Pro složité otázky proveď postupnou analýzu: obecná ustanovení → specifická ustanovení → judikatura.
- Kvalita a přesnost je důležitější než rychlost.

## POKUD NENAJDEŠ V DATABÁZI

1. Zkus jiný dotaz v `search_laws` (synonym, jiná formulace)
2. Zkus `get_full_section` s konkrétní citací (např. "89/2012 Sb.", "§ 2235")
3. Teprve pokud databáze opravdu nemá → `search_esbirka_api`
4. **NIKDY** nepoužívej `web_search` jako náhradu za právní databázi!"""


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
            "name": "search_esbirka_api",
            "description": "Přímé vyhledávání v E-Sbírka API. Použij jako ZÁLOHU když search_laws nenajde výsledek. Vrací aktuální oficiální znění ze zdroje.",
            "parameters": {
                "type": "object",
                "properties": {
                    "law_citation": {
                        "type": "string",
                        "description": "Citace zákona (např. '262/2006 Sb.', '89/2012 Sb.')"
                    },
                    "section_citation": {
                        "type": "string",
                        "description": "Volitelná citace paragrafu (např. '§ 212', '§ 2235')"
                    }
                },
                "required": ["law_citation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_law_changes",
            "description": "Získá seznam nedávných změn zákonů z E-Sbírka API. Použij pro ověření, zda byl zákon nedávno novelizován.",
            "parameters": {
                "type": "object",
                "properties": {
                    "since_days": {
                        "type": "integer",
                        "description": "Počet dní zpětně (default 30)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_section",
            "description": "Získá historickou verzi paragrafu k určitému datu. Použij pro analýzu, co zákon stanovil v minulosti (např. když se posuzuje událost z roku 2020).",
            "parameters": {
                "type": "object",
                "properties": {
                    "law_citation": {
                        "type": "string",
                        "description": "Citace zákona (např. '262/2006 Sb.')"
                    },
                    "section_citation": {
                        "type": "string",
                        "description": "Citace paragrafu (např. '§ 212')"
                    },
                    "date": {
                        "type": "string",
                        "description": "Datum ve formátu RRRR-MM-DD (např. '2020-01-15')"
                    }
                },
                "required": ["law_citation", "section_citation", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Webové vyhledávání pomocí Perplexity Sonar. Použij POUZE pro aktuální informace (čísla, data, statistiky). NEPOUŽÍVEJ jako náhradu právní databáze!",
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

# Tools subsets
TOOLS_CASE = [t for t in TOOLS if t["function"]["name"] in ["search_judgments", "get_full_judgment", "web_search"]]
TOOLS_LAW = [t for t in TOOLS if t["function"]["name"] in ["search_laws", "get_full_section", "search_esbirka_api", "get_law_changes", "get_historical_section", "web_search"]]



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
    
    async def _stream_api(
        self, 
        messages: List[Dict],
        tools_subset: Optional[List[Dict]] = None,
        use_tools: bool = True
    ) -> AsyncIterator[Dict]:
        """Call OpenRouter API with streaming."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
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
                        "tools": (tools_subset if tools_subset else TOOLS) if use_tools else None,
                        "temperature": 0.3,
                        "max_tokens": 8000,
                        "stream": True # Enable streaming
                    }
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                yield chunk
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger = logging.getLogger("agent")
            logger.error(f"API Stream failed: {e}")
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
                return json.dumps(asdict(result) if result else {"error": "Not found"}, ensure_ascii=False)
            
            elif name == "get_full_judgment":
                result = await tools.get_full_judgment(args["case_number"])
                return json.dumps(asdict(result) if result else {"error": "Not found"}, ensure_ascii=False)
            
            elif name == "web_search":
                result = await tools.web_search(args["query"])
                return json.dumps(result or {"error": "No results"}, ensure_ascii=False)
            
            elif name == "search_esbirka_api":
                result = await tools.search_esbirka_api(
                    args["law_citation"],
                    section_citation=args.get("section_citation")
                )
                return json.dumps(result or {"error": "Not found in E-Sbírka API"}, ensure_ascii=False)
            
            elif name == "get_law_changes":
                results = await tools.get_law_changes(
                    since_days=args.get("since_days", 30)
                )
                return json.dumps(results, ensure_ascii=False)
            
            elif name == "get_historical_section":
                result = await tools.get_historical_section(
                    args["law_citation"],
                    args["section_citation"],
                    args["date"]
                )
                return json.dumps(result or {"error": "Historical version not found"}, ensure_ascii=False)
            
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})
        
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def ask(
        self, 
        question: str,
        mode: str = "law",
        max_iterations: int = 20
    ) -> AsyncIterator[Dict]:
        """
        Ask a legal question with iterative tool use.
        """
        # Select tools based on mode
        if mode == "case":
            current_tools = TOOLS_CASE
        elif mode == "law":
            current_tools = TOOLS_LAW
        else:
            current_tools = TOOLS  # Default: all tools
        
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": question}
        ]
        
        # Add mode instruction
        if mode == "case":
            messages.append({
                "role": "system", 
                "content": "REŽIM: VYHLEDÁVÁNÍ JUDIKATURY. Hledej pouze v soudních rozhodnutích. Nepoužívej zákony."
            })
        
        yield {"event": "thinking", "content": "Analyzuji právní otázku..."}
        
        answer_generated = False
        force_answer_at = max_iterations - 2  # Force answer 2 iterations before limit
        
        for iteration in range(max_iterations):
            # If approaching limit, tell the LLM to generate answer NOW
            if iteration == force_answer_at:
                messages.append({
                    "role": "user", 
                    "content": "DŮLEŽITÉ: Máš ještě 2 iterace. Na základě všech informací, které jsi nashromáždil, nyní VYGENERUJ FINÁLNÍ ODPOVĚĎ. Nepoužívej další nástroje, odpověz uživateli."
                })
            # Accumulators for the stream
            full_content = ""
            tool_calls_acc = {} # index -> tool_call object
            finish_reason = None
            
            # Stream the response
            async for chunk in self._stream_api(messages, tools_subset=current_tools, use_tools=True):
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")
                
                # Handle Content
                if delta.get("content"):
                    content_chunk = delta["content"]
                    full_content += content_chunk
                    yield {"event": "answer", "content": content_chunk}
                
                # Handle Tool Calls
                if delta.get("tool_calls"):
                    for tc_chunk in delta["tool_calls"]:
                        idx = tc_chunk["index"]
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_chunk.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        
                        if "id" in tc_chunk and tc_chunk["id"]:
                            tool_calls_acc[idx]["id"] = tc_chunk["id"]
                            
                        if "function" in tc_chunk:
                            func = tc_chunk["function"]
                            if func.get("name"):
                                tool_calls_acc[idx]["function"]["name"] += func["name"]
                            if func.get("arguments"):
                                tool_calls_acc[idx]["function"]["arguments"] += func["arguments"]
            
            # Convert accumulated tool calls to list
            tool_calls = list(tool_calls_acc.values()) if tool_calls_acc else None
            
            # Construct the assistant message
            assistant_msg = {
                "role": "assistant",
                "content": full_content if full_content else None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            
            # Add to history (crucial for next turn)
            # IMPORTANT: API expects content to be string or null, not empty string if tool_calls exist
            if not assistant_msg["content"] and not assistant_msg.get("tool_calls"):
                 # Empty message? Skip
                 continue
            
            messages.append(assistant_msg)

            # ---------------------------
            # LOGIC: Tool Execution
            # ---------------------------
            if tool_calls:
                for tool_call in tool_calls:
                    func = tool_call["function"]
                    name = func["name"]
                    try:
                        args = json.loads(func["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    
                    yield {"event": "tool_call", "tool": name, "input": args}
                    
                    # Execute tool
                    result = await self._execute_tool(name, args)
                    
                    yield {"event": "tool_result", "tool": name, "result": result[:500] + "..." if len(result) > 500 else result}
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })
                # Loop continues to get next AI response
                continue
            
            # ---------------------------
            # LOGIC: Content / Length
            # ---------------------------
            if full_content:
                # If finished by length, continue generating
                if finish_reason == "length":
                     messages.append({"role": "user", "content": "Pokračuj v odpovědi přesně tam, kde jsi skončil (dokonči větu)."})
                     continue
                
                answer_generated = True
                break
            
            if finish_reason == "stop":
                 break
        
        # Fallback if no answer
        if not answer_generated and not full_content:
             yield {"event": "answer", "content": "Omlouvám se, ale nepodařilo se mi vygenerovat odpověď."}
        
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

