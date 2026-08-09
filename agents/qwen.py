from qwen_vl_utils import process_vision_info
import time

# @title inference function
def inference(processor, model, image_paths, prompt, sys_prompt="You are a helpful assistant.", max_new_tokens=4096, return_input=False, more_args={}, log_file=None, role=None, save_path=None):
    content = []
    for image_path in image_paths:
        image_local_path = "file://" + image_path
        content.append({"type": "image", "image": image_local_path})
    content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content},
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        padding=True, 
        return_tensors="pt")
    inputs = inputs.to('cuda')

    t0 = time.time()
    # truncate over-long inputs (protects against prompt > model max)
    try:
        if inputs["input_ids"].shape[1] > 7800:
            inputs["input_ids"] = inputs["input_ids"][:, :7800]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, :7800]
    except Exception:
        pass
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **more_args)
    latency_ms = (time.time() - t0) * 1000.0
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    if log_file:
        with open(log_file, "a") as f:
            for image_path in image_paths:
                f.write(image_path + "\n")
            f.write(text)
            f.write("--------------------------------------------\n")
            f.write(output_text[0])
            f.write("\n\n=============================================\n\n")

    if save_path is not None:
        try:
            from agents.stage0_utils import append_trace
            append_trace(save_path, {
                "event": "model_inference",
                "role": role or "unknown",
                "n_images": len(image_paths or []),
                "image_paths": list(image_paths or []),
                "prompt_chars": len(prompt or ""),
                "raw_output": output_text[0],
                "latency_ms": round(latency_ms, 2),
                "max_new_tokens": max_new_tokens,
                "sampled": bool(more_args),
            })
        except Exception as e:
            # Tracing must never break eval.
            if log_file:
                with open(log_file, "a") as f:
                    f.write(f"[stage0_trace_error] {e}\n")
    
    if return_input:
        return output_text[0], inputs
    else:
        return output_text[0]


def encode_image(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
