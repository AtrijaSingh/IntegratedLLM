# NOTE: Ensure that the llama-2-7b-chat.Q2_K.gguf model file is present in the 'models' directory before running this script.
import integrated_llm

import re

def clean_llama_output(text):
    # Replace token artifacts
    text = text.replace('▁', ' ')
    text = text.replace('<0x0A>', '\n')
    # Remove extra spaces
    text = re.sub(r' +', ' ', text)
    # Remove leading/trailing whitespace and repeated blank lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Optional: filter out lines that don't look like answers
    answer_lines = [line for line in lines if "capital" in line or "Paris" in line]
    if answer_lines:
        return "\n".join(answer_lines)
    else:
        return "\n".join(lines)


integrated_llm.init(r"C:/Users/atrij/Workspace/IntegratedLLM/combined_approach/NewCleanApproach_Integrated_LLM/models/llama-2-7b-chat.Q2_K.gguf")
integrated_llm.load_knowledge_file("my_faq_copy.txt")
#print(integrated_llm.query("How do I reset my product?"))

#print(integrated_llm.query("Hello"))
#print(integrated_llm.query("How long is the warranty?"))
#print(clean_llama_output(integrated_llm.query("[INST] What is the capital of France? [/INST]")))
print("-------------------------------------------------------------")
print(clean_llama_output(integrated_llm.query("[INST] Who is Atrija Singh? [/INST]")))
print("-------------------------------------------------------------")

#print(clean_llama_output(integrated_llm.query("Who is Atrija Singh?")))
#print(integrated_llm.query("What is the capital of France?"))
#print(integrated_llm.query("How long is the warranty?"))
