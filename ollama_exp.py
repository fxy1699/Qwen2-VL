from ollama import chat
res = chat(model="llama3.2-vision:latest",
                stream=False,
                messages=[
                    {
                        "role": "user",
                        "content": "你是谁"}
                ],
                options={"temperature":0}
                )
print(res)
print(res['message']['content'])