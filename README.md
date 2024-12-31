## Document Assistant with RAG

The document-assistant is a streamlit application that enables you to upload any document eg: PDF or word document, and chat with the document about its contents. You can ask specific questions to the chatbot and it will retrieve relevant information from the document uploaded and give the most appropriate response with the help of LLM(Large Language Model).

### RAG(Retrieval Augmented Generation)

The project runs on a concept called RAG(Retrieval Augmented Generation), which is nothing but a retrieval system based on context vectors and embeddings and extracting the most relevant chunk from the document with a similarity metric

![image](https://github.com/user-attachments/assets/98dba8fa-c04f-428a-84cf-8b5b5ee6892b)

### Running the application:

Please follow the below instructions to run the streamlit application:

1. You can clone the github repo with the command <code>git clone https://github.com/rajeevnair676/document-assistant.git</code>
2. Run requirements.txt with the below line in the command prompt

   <code>pip install -r requirements.txt</code>
   
3. Once the requirements are installed, you need to get an API key from Groq from the below link:
   https://console.groq.com/
   
4. Run the streamlit application with command:
   
   <code>streamlit run app.py</code>

This will open the application in browser and you can input the API key and upload a document and chat with it!!

## Future Considerations:
1. Need to include more document types
2. Working on multimodal data like images in the document uploaded
