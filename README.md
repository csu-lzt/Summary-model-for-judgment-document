# Summary-model-for-judgment-document


Judgment documents record the trial process and results of the courts, and usually contain rich legal information. However, judgment documents are characterized by long length, specialized language and complex structure. Existing automatic summary algorithms are not suitable for Chinese judgment documents. By analyzing the structural features of judgment documents and the rules of manual summary of judgment documents, we propose an Extractive-Abstractive summary model for judgment documents.  The proposed model consists of an extraction model and an abstraction model.  In the extraction model, Self-Attention mechanism is used to classify sentences into key and non-key according to whether containing important legal information. Then key sentences are extracted and combined into an initial summary. In the abstraction model, a pre-trained language model based on attention mask mechanism is used to refine the initial summary into a final summary. Such an Extractive-Abstractive model could make judgment documents more accessible to the general readers and improve the efficiency in case handling. The experimental results show that our model is better than baseline models.

### This project is an open source of our summary model, and we won the fifth place in CAIL2020 on behalf of Central South University.

![image](https://user-images.githubusercontent.com/37183558/118258305-46e5b580-b4e2-11eb-8290-7af2d6d6bc33.png)
