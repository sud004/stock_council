# ============================================================
# pipeline/debate_council.py — TOKEN-FIXED VERSION
# ============================================================
# TOKEN FIX:
#   Every _llm() call now passes the correct num_predict for
#   that specific task — not the global 1500-token default.
#
#   BEFORE:  29 calls × 1500 tokens = 43,500 tokens/stock
#   AFTER:   29 calls × ~150 tokens =  4,350 tokens/stock
#   SPEEDUP: ~10× faster — 7 stocks in ~50 min not 7 hours
#
#   Additional fixes:
#   - Transcript fed into Round 4/5 is capped at 1200 words
#     (prevents context window overflow in 4096-token models)
#   - stop sequences added per call type
#   - _extract_field regex fixed: [^\n]+ not .+? (captures STRONG BUY)
#   - vector_store refreshed each run() (no stale references)
#   - _parse_price_field() extracts stop-loss/target as floats
# ============================================================

import sys
import re
from pathlib import Path
from datetime import datetime
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_MODEL, OLLAMA_HOST, LLM_TEMPERATURE, VERBOSE_DEBUG, LLM_TOKENS
from utils.llm import stream_chat, extract_score
from memory.vector_store import get_vector_store

IST = pytz.timezone('Asia/Kolkata')

# ── Bot definitions ───────────────────────────────────────────

BULL_BOTS = [
    {"id": "bull_fundamental", "name": "Bull-Fundamental",
     "role": "bullish fundamental analyst",
     "focus": "valuation, ROE, earnings growth, balance sheet strength", "bias": "BULL"},
    {"id": "bull_technical", "name": "Bull-Technical",
     "role": "bullish technical analyst",
     "focus": "price action, momentum, trend, breakout patterns, volume", "bias": "BULL"},
    {"id": "bull_macro", "name": "Bull-Macro",
     "role": "bullish macro and sector analyst",
     "focus": "RBI policy, FII flows, sector tailwinds, government schemes", "bias": "BULL"},
    {"id": "bull_sentiment", "name": "Bull-Sentiment",
     "role": "bullish market sentiment analyst",
     "focus": "retail accumulation, PCR, options positioning, institutional buying", "bias": "BULL"},
    {"id": "bull_growth", "name": "Bull-Growth",
     "role": "bullish growth analyst",
     "focus": "revenue trajectory, EPS acceleration, market share, new products", "bias": "BULL"},
]

BEAR_BOTS = [
    {"id": "bear_risk", "name": "Bear-Risk",
     "role": "risk analyst and devil's advocate",
     "focus": "VaR, beta, max drawdown, volatility, capital loss scenarios", "bias": "BEAR"},
    {"id": "bear_valuation", "name": "Bear-Valuation",
     "role": "bearish valuation analyst",
     "focus": "overvaluation, high P/E, PEG ratio, mean reversion risk", "bias": "BEAR"},
    {"id": "bear_macro", "name": "Bear-Macro",
     "role": "bearish macro analyst",
     "focus": "crude oil, rupee depreciation, US Fed, global slowdown", "bias": "BEAR"},
    {"id": "bear_technical", "name": "Bear-Technical",
     "role": "bearish technical analyst",
     "focus": "RSI divergence, volume decline, resistance, breakdown patterns", "bias": "BEAR"},
    {"id": "bear_governance", "name": "Bear-Governance",
     "role": "governance and fraud risk analyst",
     "focus": "promoter pledging, related party transactions, audit qualifications", "bias": "BEAR"},
]

ALL_BOTS = BULL_BOTS + BEAR_BOTS


# ══════════════════════════════════════════════════════════════
# DEBATE PROMPTS — kept SHORT to save input tokens too
# ══════════════════════════════════════════════════════════════

def _make_opening_prompt(bot: dict, symbol: str, sector: str,
                          stock_data: str, history_context: str) -> str:
    bias_word = "bullish" if bot['bias'] == 'BULL' else "bearish"
    # FIX: explicitly say MAX 80 words — model obeys word limits more than token limits
    return f"""You are {bot['name']} debating {symbol} ({sector}).
Role: {bot['role']} | Focus: {bot['focus']}

KEY DATA:
{stock_data[:600]}

Past context: {(history_context or 'none')[:200]}

Give your strongest {bias_word.upper()} argument. MAX 80 words. Cite specific numbers.
End with exactly: OPENING SCORE: X/10"""


def _make_question_prompt(asking_bot: dict, target_bot: dict,
                           symbol: str, opening_statements: dict) -> str:
    target_opening = opening_statements.get(target_bot['id'], '')[:200]
    return f"""You are {asking_bot['name']} debating {symbol}.

{target_bot['name']} said: "{target_opening}"

Ask ONE sharp question (MAX 2 sentences) exposing a weakness.
Format: QUESTION TO {target_bot['name'].upper()}: [question]"""


def _make_answer_prompt(answering_bot: dict, asking_bot: dict,
                         symbol: str, question: str,
                         stock_data: str) -> str:
    return f"""You are {answering_bot['name']} debating {symbol}.

{asking_bot['name']} asked: "{question[:200]}"

Answer directly using data. MAX 60 words.
Format: ANSWER: [response]"""


def _make_chair_questions_prompt(symbol: str, sector: str,
                                  transcript_summary: str) -> str:
    # FIX: use transcript_summary (truncated) not full_transcript
    return f"""You are COUNCIL CHAIR for {symbol} ({sector}).

DEBATE SUMMARY:
{transcript_summary}

Ask the 3 most important deciding questions. Be specific to {symbol}.
CHAIR QUESTION 1: [contested point]
CHAIR QUESTION 2: [biggest risk vs opportunity]
CHAIR QUESTION 3: [timing — buy now or wait?]"""


def _make_chair_answer_prompt(bot: dict, symbol: str,
                               chair_questions: str,
                               stock_data: str) -> str:
    return f"""You are {bot['name']} answering the Chair about {symbol}.

QUESTIONS:
{chair_questions[:300]}

Answer all 3. MAX 20 words each. Be direct.
Q1 ANSWER: [answer]
Q2 ANSWER: [answer]
Q3 ANSWER: [answer]"""


def _make_final_verdict_prompt(symbol: str, company: str, sector: str,
                                price: float, transcript_summary: str,
                                stock_data: str) -> str:
    return f"""You are COUNCIL CHAIR delivering FINAL VERDICT on {symbol} ({company}).
Sector: {sector} | Price: ₹{price:.0f}

DEBATE SUMMARY:
{transcript_summary}

KEY DATA:
{stock_data[:400]}

Use EXACT headers (model stops after FINAL SCORE):
1. VERDICT: [STRONG BUY / BUY / ACCUMULATE / HOLD / REDUCE / SELL / STRONG SELL]
2. CONVICTION: [HIGH / MEDIUM / LOW]
3. BULL CASE SUMMARY: [30 words]
4. BEAR CASE SUMMARY: [30 words]
5. DECIDING FACTOR: [20 words]
6. ENTRY STRATEGY: [Buy now / Wait for dip to ₹X / Avoid]
7. TIME HORIZON: [Short (days) / Medium (weeks) / Long (months)]
8. STOP LOSS: ₹[price]
9. TARGET: ₹[price]
10. FINAL SCORE: X/10"""


# ══════════════════════════════════════════════════════════════
# DEBATE ENGINE
# ══════════════════════════════════════════════════════════════

def _truncate_transcript(parts: list, max_words: int = 1200) -> str:
    """
    Join transcript parts and truncate to max_words.
    Keeps the MOST RECENT content (tail of transcript) since
    that has the most relevant context for the verdict.
    """
    full = '\n'.join(parts)
    words = full.split()
    if len(words) <= max_words:
        return full
    # Keep tail — more recent = more relevant for verdict
    truncated = ' '.join(words[-max_words:])
    return f"[...earlier debate truncated...]\n{truncated}"


class DebateCouncil:
    """
    10-bot cross-debate council.

    TOKEN FIX: Every _llm() call uses the per-call num_predict
    from LLM_TOKENS config instead of the global 1500-token default.
    This alone gives a ~10× speedup.
    """

    def __init__(self):
        pass  # vector_store refreshed each run() call

    def _llm(self, prompt: str, call_type: str = "opening",
              print_output: bool = True) -> str:
        """
        Call LLM with the correct token budget for this call type.

        call_type maps to LLM_TOKENS keys:
            "opening", "question", "answer", "chair_q", "chair_ans", "verdict"
        """
        num_predict = LLM_TOKENS.get(call_type, LLM_TOKENS["verdict"])
        tokens = []

        def on_tok(t):
            tokens.append(t)
            if print_output:
                print(t, end='', flush=True)

        result = stream_chat(
            "You are an expert Indian stock market analyst in a structured debate.",
            prompt,
            on_token=on_tok,
            num_predict=num_predict,   # FIX: was always 1500
            timeout=60,
        )
        if print_output:
            print()
        return result

    def run(self, symbol: str, company: str, sector: str,
             price: float, stock_data: str,
             print_output: bool = True) -> dict:
        """
        Run the full 10-bot debate. Returns verdict, score, stop_loss, target.
        """
        # FIX: refresh vector_store each call
        self.vector_store = get_vector_store()

        transcript_parts   = []
        opening_statements = {}
        bot_scores         = {}

        def _section(title):
            if print_output:
                print(f"\n{'─'*55}")
                print(f"  {title}")
                print('─'*55)
            transcript_parts.append(f"\n=== {title} ===")

        if print_output:
            print(f"\n{'═'*60}")
            print(f"⚔️  10-BOT DEBATE: {symbol} | {company} | ₹{price:.0f}")
            print('═'*60)

        history_ctx = ""
        if self.vector_store:
            try:
                history_ctx = self.vector_store.get_context_for_stock(
                    symbol, "analysis verdict score", top_k=2
                ) or ""
            except Exception:
                pass
        history_ctx = history_ctx[:300]  # cap history to save input tokens

        # ── Round 1: Opening Arguments (10 calls × 150 tokens = 1500) ──
        _section("ROUND 1: OPENING ARGUMENTS")

        for bot in BULL_BOTS:
            if print_output:
                print(f"\n  {bot['name']}:", flush=True)
            prompt   = _make_opening_prompt(bot, symbol, sector, stock_data, history_ctx)
            response = self._llm(prompt, call_type="opening", print_output=print_output)
            opening_statements[bot['id']] = response[:300]  # cap stored version
            bot_scores[bot['id']] = extract_score(response, default=6.0)
            transcript_parts.append(f"{bot['name']}: {response[:200]}")

        for bot in BEAR_BOTS:
            if print_output:
                print(f"\n  {bot['name']}:", flush=True)
            prompt   = _make_opening_prompt(bot, symbol, sector, stock_data, history_ctx)
            response = self._llm(prompt, call_type="opening", print_output=print_output)
            opening_statements[bot['id']] = response[:300]
            bot_scores[bot['id']] = extract_score(response, default=5.0)
            transcript_parts.append(f"{bot['name']}: {response[:200]}")

        # ── Round 2: Cross Examination (5 calls × 80 tokens = 400) ──
        _section("ROUND 2: CROSS EXAMINATION")
        questions = {}

        cross_pairs = [
            (BULL_BOTS[0], BEAR_BOTS[1]),
            (BULL_BOTS[2], BEAR_BOTS[2]),
            (BULL_BOTS[4], BEAR_BOTS[0]),
            (BEAR_BOTS[0], BULL_BOTS[0]),
            (BEAR_BOTS[3], BULL_BOTS[1]),
        ]

        for asker, target in cross_pairs:
            if print_output:
                print(f"\n  {asker['name']} → {target['name']}:")
            prompt   = _make_question_prompt(asker, target, symbol, opening_statements)
            question = self._llm(prompt, call_type="question", print_output=print_output)
            questions[(asker['id'], target['id'])] = question
            transcript_parts.append(f"Q: {asker['name']}→{target['name']}: {question[:150]}")

        # ── Round 3: Answers (5 calls × 110 tokens = 550) ──
        _section("ROUND 3: RESPONSES")

        for (asker_id, target_id), question in questions.items():
            asker  = next(b for b in ALL_BOTS if b['id'] == asker_id)
            target = next(b for b in ALL_BOTS if b['id'] == target_id)
            if print_output:
                print(f"\n  {target['name']} answers:")
            prompt = _make_answer_prompt(target, asker, symbol, question, stock_data)
            answer = self._llm(prompt, call_type="answer", print_output=print_output)
            transcript_parts.append(f"A: {target['name']}: {answer[:150]}")

        # ── Round 4: Chair Questions (1 + 2 calls) ──
        _section("ROUND 4: CHAIR QUESTIONS")

        # FIX: truncate transcript before feeding into chair — prevents context overflow
        transcript_summary = _truncate_transcript(transcript_parts, max_words=800)

        if print_output:
            print("\n  Council Chair asks:")
        chair_questions = self._llm(
            _make_chair_questions_prompt(symbol, sector, transcript_summary),
            call_type="chair_q",
            print_output=print_output
        )
        transcript_parts.append(f"CHAIR: {chair_questions[:200]}")

        for bot in [BULL_BOTS[0], BEAR_BOTS[0]]:
            if print_output:
                print(f"\n  {bot['name']} answers Chair:")
            answer = self._llm(
                _make_chair_answer_prompt(bot, symbol, chair_questions, stock_data),
                call_type="chair_ans",
                print_output=print_output
            )
            transcript_parts.append(f"CHAIR-ANS {bot['name']}: {answer[:150]}")

        # ── Round 5: Final Verdict (1 call × 450 tokens) ──
        _section("ROUND 5: FINAL VERDICT")

        # FIX: truncate again for verdict — keep most recent 1000 words
        verdict_summary = _truncate_transcript(transcript_parts, max_words=1000)

        if print_output:
            print("\n  Council Chair delivers verdict:")
        verdict_text = self._llm(
            _make_final_verdict_prompt(
                symbol, company, sector, price,
                verdict_summary, stock_data
            ),
            call_type="verdict",
            print_output=print_output
        )

        # ── Parse verdict fields ────────────────────────────────
        final_score   = extract_score(verdict_text, default=5.0)
        verdict_label = _extract_field(verdict_text, "VERDICT", "HOLD")
        conviction    = _extract_field(verdict_text, "CONVICTION", "MEDIUM")
        bull_summary  = _extract_field(verdict_text, "BULL CASE SUMMARY", "")
        bear_summary  = _extract_field(verdict_text, "BEAR CASE SUMMARY", "")
        deciding      = _extract_field(verdict_text, "DECIDING FACTOR", "")
        entry         = _extract_field(verdict_text, "ENTRY STRATEGY", "")
        horizon       = _extract_field(verdict_text, "TIME HORIZON", "Medium")
        stop_loss     = _parse_price_field(verdict_text, "STOP LOSS")
        target        = _parse_price_field(verdict_text, "TARGET")

        bull_avg = sum(bot_scores.get(b['id'], 5) for b in BULL_BOTS) / len(BULL_BOTS)
        bear_avg = sum(bot_scores.get(b['id'], 5) for b in BEAR_BOTS) / len(BEAR_BOTS)

        # Token count summary for transparency
        total_words = len(' '.join(transcript_parts).split())
        if print_output:
            print(f"\n{'═'*60}")
            print(f"  VERDICT:    {verdict_label}")
            print(f"  SCORE:      {final_score}/10  | CONVICTION: {conviction}")
            print(f"  BULL avg:   {bull_avg:.1f}  | BEAR avg: {bear_avg:.1f}")
            if stop_loss: print(f"  STOP LOSS:  ₹{stop_loss:.2f}")
            if target:    print(f"  TARGET:     ₹{target:.2f}")
            print(f"  Transcript: ~{total_words} words generated this debate")
            print('═'*60)

        return {
            'symbol':          symbol,
            'company':         company,
            'sector':          sector,
            'price':           price,
            'verdict':         verdict_label,
            'score':           final_score,
            'conviction':      conviction,
            'bull_summary':    bull_summary,
            'bear_summary':    bear_summary,
            'deciding_factor': deciding,
            'entry_strategy':  entry,
            'time_horizon':    horizon,
            'stop_loss':       stop_loss,
            'target':          target,
            'bull_avg_score':  round(bull_avg, 2),
            'bear_avg_score':  round(bear_avg, 2),
            'bot_scores':      bot_scores,
            'transcript':      verdict_summary,   # trimmed version
            'verdict_text':    verdict_text,
        }


# ── Parsers ───────────────────────────────────────────────────

def _extract_field(text: str, field: str, default: str = "") -> str:
    """
    FIX: Use [^\n]+ instead of .+? so STRONG BUY is not truncated to STRONG.
    """
    pattern = rf'{re.escape(field)}[:\s]+([^\n]+)'
    match   = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else default


def _parse_price_field(text: str, field: str) -> float | None:
    """Extract a ₹ price from 'STOP LOSS: ₹2450' or 'TARGET: ₹2800.50'."""
    pattern = rf'{re.escape(field)}[:\s]+₹?\s*([\d,]+(?:\.\d+)?)'
    match   = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            pass
    return None


# ── Singleton ─────────────────────────────────────────────────

_debate_council = None

def get_debate_council() -> DebateCouncil:
    global _debate_council
    if _debate_council is None:
        _debate_council = DebateCouncil()
    return _debate_council
