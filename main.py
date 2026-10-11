import json
import onnxruntime_genai as onnx
from subprocess import run

model_path = "C:/src/models/qwen_coder_cpu"

system_instruction = (
    "You are an expert at writing git commit messages. You will provide useful commit messages based on the user's provided git diffs. "
    "You must follow the conventional commits format (e.g. feat(scope):desc). "
    "You must use an imperative tone and respond in the present tense. "
    "You must not include anything other than the commit message. "
    "You must not attempt to explain the provided code. "
    "You must not generate any code."
)
def main():
    result = run(["git", "diff"], check=True, capture_output=True)
    git_diff = result.stdout
    model = onnx.Model(model_path)
    tokenizer = onnx.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"<git_diff>\n{git_diff.strip()}\n</git_diff>"}
    ]

    formatted_prompt = tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)
    input_tokens = tokenizer.encode(formatted_prompt)

    parameters = onnx.GeneratorParams(model)
    parameters.set_search_options(
        max_length=1024
    )

    generator = onnx.Generator(model, parameters)
    generator.append_tokens(input_tokens)

    try:
        while not generator.is_done():
            generator.generate_next_token()
            token_id = generator.get_next_tokens()

            decoded_char = tokenizer_stream.decode(token_id[0])
            print(decoded_char, end="", flush=True)
    finally:
        del generator
    print()

    return 0


if __name__ == "__main__":
    main()
