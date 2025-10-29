import asyncio
import argparse
import json
from datasets import load_dataset
from openai import AsyncOpenAI
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


# Instruction templates. Add your own template for your HuggingFace dataset if applicable
TEMPLATES = {
    "gsm8k": "Solve this math problem step by step:\n\n{question}",
    "alpaca": "{instruction}",
    "squad": "Answer the following question based on the context.\n\nContext: {context}\n\nQuestion: {question}",
    "mmlu": "Answer the following multiple choice question:\n\n{question}",
    "tower": "Translate the following {source_lang} source text to {target_lang}:\n{source_lang}: {text}\n{target_lang}: ",
    "default": "{text}",
}

def parser_args():
    parser = argparse.ArgumentParser(
        description="vLLM inference on HuggingFace datasets"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--dataset", type=str, required=True, help="HuggingFace dataset name"
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--subset", type=str, default="main", help="Dataset subset")
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server URL",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Max tokens to generate"
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=100, help="Max concurrent requests"
    )
    parser.add_argument(
        "--instruction_template",
        type=str,
        default=None,
        help="Instruction template with {field_name} placeholders (e.g., 'Solve: {question}')",
    )
    parser.add_argument(
        "--template_preset",
        type=str,
        choices=list(TEMPLATES.keys()),
        default=None,
        help="Use a preset template",
    )
    parser.add_argument(
        "--output", type=str, default="predictions.json", help="Output file"
    )

    return parser.parse_args()

def format_prompt(item, template: str):
    """Format item using template string with field names in curly braces"""
    return template.format(**item)


async def generate_predictions(
    dataset,
    model_name: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    instruction_template: str,
    max_concurrent: int = 100,
):
    client = AsyncOpenAI(base_url=base_url, api_key="not necessary for vLLM")

    async def process_item(item, idx):
        try:
            prompt = format_prompt(item, instruction_template)
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {
                "idx": idx,
                "prompt": prompt,
                "response": response.choices[0].message.content,
                **item,
            }
        except Exception as e:
            logging.warning(f"[{idx}] Failed to process item: {e}")
            return {"idx": idx, "error": str(e)}

    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_process(item, idx):
        async with semaphore:
            return await process_item(item, idx)

    tasks = [bounded_process(item, idx) for idx, item in enumerate(dataset)]
    results = await asyncio.gather(*tasks)

    return results


def main():
    args = parser_args()
    # Determine template
    if args.instruction_template:
        template = args.instruction_template
    elif args.template_preset:
        template = TEMPLATES[args.template_preset]
    else:
        # Try to infer from dataset name
        for key in TEMPLATES:
            if key in args.dataset.lower():
                template = TEMPLATES[key]
                print(f"Auto-detected template: {key}")
                break
        else:
            template = TEMPLATES["default"]
            print(
                f"Using default template. Consider using --instruction_template or --template_preset"
            )

    print(f"Template: {template}")

    # Load dataset
    dataset = load_dataset(args.dataset, args.subset, split=args.split)

    # Run inference
    results = asyncio.run(
        generate_predictions(
            dataset,
            args.model,
            args.base_url,
            args.max_tokens,
            args.temperature,
            template,
            args.max_concurrent,
        )
    )

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} predictions, saved to {args.output}")


if __name__ == "__main__":
    main()
