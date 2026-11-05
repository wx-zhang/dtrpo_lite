"""LiveCodeBench release_v6 prompts, batched generation and pass@1 scoring."""
import json

from eval.common import write_json


def build_prompt(problem, tokenizer):
    prompt = (
        "You will be given a question (problem specification) and will generate "
        "a correct Python program that matches the specification and passes all "
        "tests.\n\n"
        f"Question: {problem.question_content}\n\n"
    )
    if problem.starter_code:
        prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        prompt += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        prompt += f"Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def extract_code(text):
    """Keep the last fenced code block, matching the LCB Fast-dLLM protocol."""
    lines = text.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    return "\n".join(lines[fences[-2] + 1:fences[-1]]) if len(fences) >= 2 else ""


def run(generator, job, directory):
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    dataset = load_code_generation_dataset("release_v6")
    if job["limit"] is not None:
        dataset = dataset[:job["limit"]]
    c = job["config"]
    requests = [(i, build_prompt(p, generator.tokenizer)) for i, p in enumerate(dataset)]
    requests.sort(key=lambda r: len(generator.tokenizer(r[1]).input_ids))
    completed = {}
    kwargs = {k: v for k, v in c.items() if k != "seed"}
    with (directory / "raw_generations.jsonl").open("w") as f:
        for start in range(0, len(requests), c["batch_size"]):
            chunk = requests[start:start + c["batch_size"]]
            outputs = generator.batch_generate([r[1] for r in chunk], **kwargs)
            for (index, prompt), text in zip(chunk, outputs):
                row = dict(index=index, question_id=dataset[index].question_id, prompt=prompt,
                           generation=text, code=extract_code(text))
                completed[index] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            print(f"LCB {len(completed)}/{len(dataset)}", flush=True)
    codes = [[completed[i]["code"]] for i in range(len(dataset))]
    write_json(directory / "lcb_outputs.json", [dict(question_id=p.question_id, code_list=code)
               for p, code in zip(dataset, codes)])
    metrics = codegen_metrics([p.get_evaluation_sample() for p in dataset], codes,
                              num_process_evaluate=job["score_workers"])
    write_json(directory / "metrics.json", metrics)
    return dict(score=metrics[0]["pass@1"], metric="pass@1", n=len(dataset))
