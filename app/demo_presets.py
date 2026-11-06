"""Curated demonstration inputs and validated request settings, independent of local runs."""

from app.schemas.comparison import CompareRequest
from app.schemas.routing import Constraints, RouteRequest

# Fixed benchmark examples make prompt-aware and historical replay directly comparable.
PROMPTS = {
    "sum_001": (
        "Summarize the following text in 2-3 sentences:\n"
        "\n"
        "Artificial intelligence (AI) has transformed numerous industries, from healthcare "
        "diagnostics to financial fraud detection. Machine learning algorithms can now "
        "process vast amounts of data to identify patterns that would take humans months to "
        "discover. However, the rapid adoption of AI also raises important questions about "
        "bias, transparency, and accountability. Organizations must carefully consider these "
        "ethical dimensions alongside the technical capabilities when deploying AI systems. "
        "The balance between innovation and responsible deployment remains one of the most "
        "critical challenges facing the technology sector today."
    ),
    "sum_002": (
        "Summarize the following text in 2-3 sentences:\n"
        "\n"
        "Cloud computing has fundamentally changed how businesses manage their IT "
        "infrastructure. Rather than maintaining expensive on-premises servers, companies can "
        "now rent computing resources on demand from providers like AWS, Google Cloud, and "
        "Microsoft Azure. This shift has enabled startups to launch products without massive "
        "upfront capital investment. The pay-as-you-go model allows organizations to scale "
        "resources up or down based on actual usage, resulting in significant cost savings. "
        "However, concerns about vendor lock-in, data sovereignty, and security continue to "
        "influence enterprise cloud adoption strategies."
    ),
    "sum_003": (
        "Summarize the following text in 2-3 sentences:\n"
        "\n"
        "Quantum computing represents a paradigm shift in computational power. Unlike "
        "classical computers that process information in binary bits (0 or 1), quantum "
        "computers use qubits that can exist in multiple states simultaneously through "
        "superposition. This enables quantum machines to solve certain types of problems "
        "exponentially faster than classical computers. Current applications include drug "
        "discovery, cryptography, and optimization problems. Despite significant progress by "
        "companies like IBM and Google, practical large-scale quantum computing still faces "
        "challenges including qubit stability and error correction."
    ),
    "sum_004": (
        "Summarize in 2 sentences: A team cut cloud costs by removing idle servers. It "
        "retained backups and monitored performance to protect reliability."
    ),
    "cls_001": (
        "Classify the sentiment of the following customer review as POSITIVE, NEGATIVE, or "
        "NEUTRAL. Respond with only the classification label.\n"
        "\n"
        'Review: "I absolutely love this product! It arrived on time, the packaging was '
        "beautiful, and the quality exceeded my expectations. I've already recommended it to "
        "all my friends. Best purchase I've made this year!\""
    ),
    "cls_002": (
        "Classify the sentiment of the following customer review as POSITIVE, NEGATIVE, or "
        "NEUTRAL. Respond with only the classification label.\n"
        "\n"
        'Review: "The product is okay. It does what it says but nothing special. Delivery was '
        "on time. I might buy it again if there's no better alternative, but I wouldn't go "
        'out of my way to recommend it."'
    ),
    "cls_003": (
        "Classify the sentiment of the following customer review as POSITIVE, NEGATIVE, or "
        "NEUTRAL. Respond with only the classification label.\n"
        "\n"
        'Review: "Terrible experience. The item arrived damaged and customer support was '
        "unhelpful. I waited three weeks for a replacement that never came. Finally got a "
        'refund but wasted so much time. Would never buy from this company again."'
    ),
    "ext_001": (
        "Extract the following information from the text below and return it as JSON with "
        "keys: company_name, revenue, year, growth_rate.\n"
        "\n"
        'Text: "TechVision Corp reported annual revenue of $4.2 billion for fiscal year 2025, '
        "representing a 15.3% increase compared to the previous year. The company attributed "
        'its strong performance to growth in its cloud services division."'
    ),
    "ext_002": (
        "Extract the following information from the text below and return it as JSON with "
        "keys: person_name, job_title, company, location.\n"
        "\n"
        'Text: "Sarah Chen has been appointed as the new Chief Technology Officer at DataFlow '
        "Systems, a leading analytics firm headquartered in San Francisco. She previously "
        'served as VP of Engineering at CloudScale Technologies."'
    ),
    "ext_004": (
        "Return only JSON with keys product and quantity. Text: The order contains 12 keyboards."
    ),
    "rea_001": (
        "A service spends $120 per month. Optimization reduces spend by 25 percent. What is "
        "the new monthly spend? Explain briefly."
    ),
    "rea_002": (
        "Each request uses 200 input tokens and 100 output tokens. How many total tokens do "
        "10 requests use? Explain briefly."
    ),
}


def _route(
    prompt: str,
    *,
    mode: str = "prompt_aware",
    policy: str = "quality_first",
    quality: float = 0.5,
    cost: float = 0.01,
    max_tokens: int = 1024,
) -> RouteRequest:
    return RouteRequest(
        prompt=prompt,
        task_type="auto",
        routing_mode=mode,
        policy=policy,
        constraints=Constraints(
            min_quality=quality,
            max_cost=cost,
            max_latency_ms=180000,
            min_reliability=0.8,
        ),
        max_tokens=max_tokens,
        execute=False,
    )


COMPARISON_PRESETS: dict[str, tuple[str, CompareRequest]] = {
    "Cloud infrastructure summary": (
        "Compare summary content, response time, token usage, and estimated cost.",
        CompareRequest(prompt=PROMPTS["sum_002"], task_type="summarization"),
    ),
    "Positive customer sentiment": (
        "Compare label accuracy and whether each model follows the label-only instruction.",
        CompareRequest(prompt=PROMPTS["cls_001"], task_type="classification"),
    ),
    "AI and responsible deployment": (
        "Compare how models retain the passage's technical and ethical themes.",
        CompareRequest(prompt=PROMPTS["sum_001"], task_type="summarization"),
    ),
    "Neutral customer sentiment": (
        "Test a mixed review whose reference sentiment is NEUTRAL.",
        CompareRequest(prompt=PROMPTS["cls_002"], task_type="classification", max_tokens=256),
    ),
    "Negative customer sentiment": (
        "Test a complaint whose reference sentiment is NEGATIVE.",
        CompareRequest(prompt=PROMPTS["cls_003"], task_type="classification", max_tokens=256),
    ),
    "Financial fields as JSON": (
        "Compare structured extraction of company, revenue, year, and growth rate.",
        CompareRequest(prompt=PROMPTS["ext_001"], task_type="extraction", max_tokens=512),
    ),
    "Appointment details as JSON": (
        "Compare extraction of a person's name, role, company, and location.",
        CompareRequest(prompt=PROMPTS["ext_002"], task_type="extraction", max_tokens=512),
    ),
    "Order quantity as JSON": (
        "Check whether models preserve a numeric quantity and return only the requested keys.",
        CompareRequest(prompt=PROMPTS["ext_004"], task_type="extraction", max_tokens=256),
    ),
    "Monthly spend calculation": (
        "Compare arithmetic and explanations for a 25 percent reduction from $120.",
        CompareRequest(prompt=PROMPTS["rea_001"], task_type="reasoning", max_tokens=512),
    ),
    "Token usage calculation": (
        "Compare reasoning about input and output usage across ten requests.",
        CompareRequest(prompt=PROMPTS["rea_002"], task_type="reasoning", max_tokens=512),
    ),
    "Markdown summary preview": (
        "Demonstrate rendered headings, emphasis, and bullet lists in the answer preview.",
        CompareRequest(
            prompt="Summarize this passage as Markdown with a heading and three bullet points.\n\n"
            + PROMPTS["sum_002"].split("\n\n", 1)[1],
            task_type="summarization",
            max_tokens=512,
        ),
    ),
}


ROUTING_PRESETS: dict[str, tuple[str, RouteRequest]] = {
    "Prompt-aware / AI summary": (
        "Start the four-summary replay here. Quality-first settings stay identical across topics.",
        _route(PROMPTS["sum_001"]),
    ),
    "Prompt-aware / Cloud summary": (
        "Compare this topic's matched benchmark evidence with the AI summary.",
        _route(PROMPTS["sum_002"]),
    ),
    "Prompt-aware / Quantum summary": (
        "Inspect the quantum passage's evidence and its quality-first model selection.",
        _route(PROMPTS["sum_003"]),
    ),
    "Prompt-aware / Short server-cost summary": (
        "Compare a short summary with the longer passages using the same requirements and policy.",
        _route(PROMPTS["sum_004"]),
    ),
    "Prompt-aware / Classification quality first": (
        "Demonstrate automatic task detection and selection using sentiment benchmark evidence.",
        _route(PROMPTS["cls_001"]),
    ),
    "Prompt-aware / Extraction quality first": (
        "Inspect evidence for extracting appointment details as JSON.",
        _route(PROMPTS["ext_002"]),
    ),
    "Prompt-aware / Reasoning balanced": (
        "Balance quality, cost, latency, and reliability for an arithmetic request.",
        _route(PROMPTS["rea_001"], policy="balanced"),
    ),
    "Prompt-aware / Cloud cost first": (
        "Keep the cloud prompt fixed and compare cost-first with its quality-first preset.",
        _route(PROMPTS["sum_002"], policy="cost_first"),
    ),
    "Prompt-aware / Short summary latency first": (
        "Keep the short server-cost prompt fixed and emphasize response time.",
        _route(PROMPTS["sum_004"], policy="latency_first"),
    ),
    "Prompt-aware / Cloud with a 64-token budget": (
        "Reserve 64 output tokens under a $0.0005 cost limit. Compare the larger-budget preset.",
        _route(PROMPTS["sum_002"], cost=0.0005, max_tokens=64),
    ),
    "Prompt-aware / Cloud with a 2048-token budget": (
        "Reserve more output at the same $0.0005 cost limit. Inspect model rejection reasons.",
        _route(PROMPTS["sum_002"], cost=0.0005, max_tokens=2048),
    ),
    "Prompt-aware / Unfamiliar topic": (
        "Inspect task-prior fallback when a summary has no close benchmark topic match.",
        _route("Summarize in 2 lines: Superconducting vortices form ordered lattices.", quality=0),
    ),
    "Prompt-aware / Zero cost limit": (
        "Inspect cost-constraint failures when estimated per-request cost must be zero.",
        _route(PROMPTS["sum_002"], quality=0, cost=0),
    ),
    "Historical / AI summary": (
        "Replay the AI summary using saved task averages and compare with its prompt-aware preset.",
        _route(PROMPTS["sum_001"], mode="task_average"),
    ),
    "Historical / Cloud summary": (
        "Keep policy and requirements fixed. Task-average metrics are shared across summaries.",
        _route(PROMPTS["sum_002"], mode="task_average"),
    ),
    "Historical / Quantum summary": (
        "Compare the historical summary metrics and ranking with the other summary topics.",
        _route(PROMPTS["sum_003"], mode="task_average"),
    ),
    "Historical / Short server-cost summary": (
        "Contrast the shared summarization prior with the prompt-aware short-summary estimate.",
        _route(PROMPTS["sum_004"], mode="task_average"),
    ),
    "Historical / Classification quality first": (
        "Prioritize measured classification quality under the historical task-average basis.",
        _route(PROMPTS["cls_001"], mode="task_average"),
    ),
    "Historical / Classification cost first": (
        "Keep the sentiment prompt fixed and compare the effect of the cost-first policy.",
        _route(PROMPTS["cls_001"], mode="task_average", policy="cost_first"),
    ),
    "Historical / Extraction quality first": (
        "Use measured extraction averages and compare with prompt-aware appointment extraction.",
        _route(PROMPTS["ext_002"], mode="task_average"),
    ),
    "Historical / Reasoning balanced": (
        "Inspect reasoning selection based on measured task averages.",
        _route(PROMPTS["rea_001"], mode="task_average", policy="balanced"),
    ),
    "Historical / Zero cost limit": (
        "Apply a zero cost limit to measured average costs and inspect the failed constraints.",
        _route(PROMPTS["sum_002"], mode="task_average", quality=0, cost=0),
    ),
}
