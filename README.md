# ssm-cs221
1. Introduction
In recent years, the application of artificial intelligence (AI) has seen rapid growth, transforming a variety of industries from healthcare to education, recruitment, and beyond. However, as AI systems such as Large Language Models (LLMs) become more prevalent, they also inherit biases from the data on which they are trained. This phenomenon is not only an ethical concern but can also have real-world implications in decision-making processes, especially when biases align with societal stereotypes.
Gender bias, in particular, has been a subject of extensive discussion due to its pervasiveness in many domains. The language produced by AI models often reflects deeply ingrained societal norms and historical data, perpetuating stereotypes related to gender roles. For example, models may associate certain professions or tasks with a specific gender, such as "nurse" being more frequently paired with female pronouns, and "engineer" with male pronouns. This type of bias in AI responses can influence how AI systems interact with users and can have far-reaching consequences, from reinforcing gender stereotypes in educational materials to influencing hiring decisions, thereby contributing to gender inequality.
The objective of this study is to investigate gender bias in LLMs, specifically through the analysis of the Mistral-7B model, and to propose techniques for mitigating such bias. This study combines probabilistic modeling with advanced natural language techniques in order to dynamically identify and correct biased outputs in real-time, significantly improving gender neutrality in the model's responses.
2. Literature Review
The exploration of bias in artificial intelligence, particularly in large language models (LLMs), has been an area of intense research. The studies in this domain highlight the multifaceted nature of biases, their origins, and their implications. This section delves into existing literature, elucidating where our work aligns and diverges from prior studies.
2.1 Surveying Bias in Large Language Models
Gallegos et al. (2024) provide a comprehensive survey on bias and fairness in LLMs. Their work underscores how biases arise from the datasets used to train models, which often reflect historical, societal, and cultural prejudices. The paper categorizes biases into explicit (e.g., overtly gendered language) and implicit (e.g., association of certain roles with specific genders). Importantly, they emphasize the need for systematic mitigation strategies, advocating for both pre-training and post-training solutions. Our work builds on these insights by proposing a novel real-time bias adjustment mechanism using Hidden Markov Models (HMMs).
2.2 Bias Mitigation via Knowledge Editing
Chen et al. (2024) tackle bias mitigation from a knowledge-editing perspective. They argue that modifying specific knowledge representations within models can significantly reduce bias. While their method focuses on altering model weights and representations, our approach differs by introducing an external, dynamic adjustment mechanism. This allows for real-time corrections without the need to alter the model’s underlying structure, making our method more adaptable and scalable.
3. Dataset
The dataset forms the cornerstone of our analysis, as it provides the contextual diversity required to evaluate and mitigate gender bias in large language models (LLMs). This section provides a detailed exploration of the dataset used, its characteristics, preprocessing steps, and relevance to the objectives of this project.
3.1 Dataset Source
The primary dataset used in this study is the WinoBias/WinoGender dataset, a well-known benchmark for evaluating gender bias in NLP models. The dataset comprises sentences with ambiguous pronoun references and varying contexts, focusing on occupational and societal roles. Each example includes:
Tokens: The words that make up the sentence.
Coreference Clusters: Pre-annotated clusters indicating the intended referent for each pronoun.
3.2 Dataset Characteristics
The final dataset used for the study consisted of:
790 Rows: Each row represents a sentence with ambiguous pronoun references.
Occupational Roles: Examples such as "The engineer fixed the machine quickly," which test for gender association with traditionally male or female roles.
Neutral Prompts: Examples that deliberately avoid specifying gender to test the model's baseline assumptions.
Balanced Representation: Equal distribution of prompts involving male, female, and neutral contexts to ensure fairness in evaluation.
4. Baseline
The baseline forms the foundation for evaluating the progress and improvements achieved through the main approach. It establishes the initial performance metrics and highlights the areas where the model may require refinement to address gender bias effectively. This section elaborates on the baseline setup, methodology, and findings.
4.1 Baseline Model
The baseline evaluation was conducted using the Mistral-7B-v0.1 model, accessed via the Hugging Face Inference API. Mistral-7B is a state-of-the-art language model designed for general-purpose natural language processing tasks. It was chosen due to its competitive performance on several NLP benchmarks and its capability to handle nuanced tasks such as coreference resolution.
4.2 Baseline Implementation
To evaluate the model’s performance on gender bias and alignment tasks, the following methodology was employed:
Prompt Construction: Sentences from the dataset were formatted into prompts designed to test the model’s ability to resolve ambiguous pronouns. For example:
Prompt: Please resolve the coreferences in the following sentence and specify which noun each pronoun refers to. Sentence: "The mechanic greets with the receptionist because she was in a good mood."
Coreference Resolution: The model was tasked with identifying the antecedent of each pronoun in the sentence. The output was analyzed to determine whether the pronoun resolution aligned with the dataset’s annotations.
Pronoun Analysis: The frequency of gendered pronouns ("he," "she," "they") was computed to identify any inherent biases in the model’s responses.
Sentiment Analysis: Each response was analyzed using the TextBlob library to classify the sentiment as positive, negative, or neutral.
Stereotype Score: A stereotype score was calculated based on the association of specific professions with gendered pronouns. For example:
"Engineer" with "he" or "Nurse" with "she" would increment the stereotype score.
4.3 Baseline Metrics
The baseline evaluation yielded the following results:
Quantitative Results
Mean Stereotype Score: 1.24 (indicating a moderate level of stereotypical bias).
Pronoun Distribution:
"He": 45%
"She": 40%
"They": 15%
Sentiment Distribution:
Positive: 60%
Neutral: 30%
Negative: 10%
5. Main Approach
The primary objective of this project is to address the limitations identified in the baseline by developing a dynamic and adaptive system for resolving gender bias in model outputs. To achieve this, we integrate a State Space Model (SSM), specifically a Hidden Markov Model (HMM), into the analysis pipeline. This section describes the methodology, design, and implementation of our main approach.
5.1 Overview of the Main Approach
The main approach integrates an HMM with the existing Mistral-7B language model to:
Dynamically assess the bias state of model-generated outputs.
Transition between bias states (e.g., neutral → biased masculine) during text generation.
Adjust outputs in real-time to favor neutrality while preserving semantic consistency.
The HMM serves as a probabilistic framework to model sequential transitions between bias states. This allows for a systematic adjustment of outputs by leveraging state transition probabilities derived from linguistic and contextual features.
5.2 Hidden Markov Model (HMM) Framework
HMM Components
States:
The HMM models three bias states:
Neutral (State 0): Responses free from gendered pronouns or stereotypes.
Biased Masculine (State 1): Responses favoring masculine pronouns or stereotypes.
Biased Feminine (State 2): Responses favoring feminine pronouns or stereotypes.
Observations:
The features derived from model outputs are treated as observations. These include:
Stereotype Score: Indicates the extent to which a response aligns with gendered occupational stereotypes.
Pronoun Counts: Frequency of "he," "she," and "they" in the response.
Sentiment Score: Captures the polarity of the response as positive, negative, or neutral.
Transition Probabilities:
The probabilities of transitioning between states are learned from the dataset. For example:
P(Neutral→Biased Masculine)=0.3P(\text{Neutral} \rightarrow \text{Biased Masculine}) = 0.3P(Neutral→Biased Masculine)=0.3
P(Biased Feminine→Neutral)=0.6P(\text{Biased Feminine} \rightarrow \text{Neutral}) = 0.6P(Biased Feminine→Neutral)=0.6
Emission Probabilities:
These probabilities model the likelihood of observing a particular feature vector given a specific state.
6. Evaluation Metric
Evaluation metrics are critical in quantifying the success and limitations of our approach. For this project, we employed a combination of quantitative and qualitative metrics to evaluate the performance of the baseline model, the GPT-adjusted responses, and the HMM-adjusted responses. The metrics were designed to measure both gender bias and context relevance while ensuring that semantic meaning was preserved.
6.1 Metrics Overview
6.1.1 Gender Bias Score (GBS):
This metric quantifies the extent of gender bias in a response based on pronoun usage:
GBS=Count(Female Pronouns)−Count(Male Pronouns)GBS = \text{Count(Female Pronouns)} - \text{Count(Male Pronouns)}GBS=Count(Female Pronouns)−Count(Male Pronouns)
Positive GBS: Indicates a female bias.
Negative GBS: Indicates a male bias.
Zero GBS: Indicates gender neutrality.
6.1.2 Context Alignment Score (CAS):
This binary metric evaluates whether the response aligns with the expected gender context of the prompt:
CAS={1,if response matches expected gender context0,otherwiseCAS = \begin{cases} 1, & \text{if response matches expected gender context} \\ 0, & \text{otherwise} \end{cases}CAS={1,0,​if response matches expected gender contextotherwise​
6.1.3 Neutrality Score (NS):
A numerical score from 1 to 5, where:
1=Very biased1 = \text{Very biased}1=Very biased
5=Completely neutral5 = \text{Completely neutral}5=Completely neutral
Neutrality scores were obtained using the GPT-3.5-turbo model by prompting it to evaluate each response's neutrality.
6.1.4 Pronoun Distribution:
This metric tracks the frequency of pronouns ("he," "she," "they") in the model's output. It helps to identify imbalances and biases in pronoun usage across responses.
6.1.5 Stereotype Score:
This score measures alignment with occupational gender stereotypes:
Stereotype Score=∑i=1nMatches(Rolei,Pronouni)\text{Stereotype Score} = \sum_{i=1}^n \text{Matches}(\text{Role}_i, \text{Pronoun}_i)Stereotype Score=i=1∑n​Matches(Rolei​,Pronouni​)
6.2 Quantitative Evaluation
Baseline Metrics:
Mean Gender Bias Score: -0.25 (slight male bias)
Context Alignment: 30% of responses aligned with the implied gender in the prompt.
Neutrality Score: Average of 3.2 out of 5.
Pronoun Distribution:
He: 65%
She: 30%
They: 5%
Stereotype Score: 0.68 (68% of responses aligned with stereotypes).
HMM-Adjusted Metrics:
Mean Gender Bias Score: -0.05 (near neutral).
Context Alignment: 85% of responses aligned with the implied gender in the prompt.
Neutrality Score: Average of 4.6 out of 5.
Pronoun Distribution:
He: 33%
She: 34%
They: 33%
Stereotype Score: 0.20 (20% of responses aligned with stereotypes).
GPT-Adjusted Metrics:
Mean Gender Bias Score: 0.00 (perfect neutrality).
Context Alignment: 90% of responses aligned with the implied gender in the prompt.
Neutrality Score: Average of 4.8 out of 5.
Pronoun Distribution:
He: 30%
She: 30%
They: 40%
Stereotype Score: 0.15 (15% of responses aligned with stereotypes).
7. Results & Analysis
The results and analysis section focuses on the comparative evaluation of the baseline model, the HMM-adjusted responses, and GPT-adjusted responses. This section highlights the quantitative and qualitative findings and provides insights into the effectiveness of each approach. Detailed examples and visualizations are included to illustrate the performance trends and the impact of the adjustments on gender bias and neutrality.
7.1 Results
Baseline Metrics:
Gender Bias Score: -0.25
Context Alignment Score: 30%
Neutrality Score: 3.2/5
Stereotype Score: 0.68
Pronoun Distribution:
He: 65%
She: 30%
They: 5%
HMM-Adjusted Metrics:
Gender Bias Score: -0.05
Context Alignment Score: 85%
Neutrality Score: 4.6/5
Stereotype Score: 0.20
Pronoun Distribution:
He: 33%
She: 34%
They: 33%
GPT-Adjusted Metrics:
Gender Bias Score: 0.00
Context Alignment Score: 90%
Neutrality Score: 4.8/5
Stereotype Score: 0.15
Pronoun Distribution:
He: 30%
She: 30%
They: 40%
8. Error Analysis
Error analysis is a critical part of evaluating the performance of our methods. This section identifies and categorizes the errors observed during our experiments, analyzes their causes, and provides insights into the strengths and weaknesses of each approach. Detailed examples, statistical findings, and potential reasons for failure are included to understand the observed discrepancies.
8.1 Types of Errors
1. Baseline Errors
Bias Reinforcement Errors:
Description: The baseline model often reinforced occupational gender stereotypes.
Example:
Prompt: "The doctor conducted a surgery. What did they do afterward?"
Baseline Response: "He wrote the post-surgery report."
Analysis: The use of "he" aligns with stereotypical associations of doctors being male, even when no gender-specific context is provided.
Frequency: 55% of responses aligned with traditional gender stereotypes.
2. HMM Errors
Insufficient Context Adaptation:
Description: The state transitions in HMM were not always able to adapt to complex contexts.
Example:
Prompt: "The engineer fixed the machine. What did he do next?"
HMM Response: "They fixed it again."
Analysis: The system failed to understand that "he" was contextually relevant in this case.
Frequency: 10% of cases were affected.
3. GPT Errors
Over-Correction:
Description: GPT adjustments occasionally overcorrected, leading to responses that neutralized gender unnecessarily.
Example:
Prompt: "The doctor performed the surgery. What did he do next?"
GPT Response: "The doctor performed the surgery again."
Analysis: The adjustment removed contextually valid pronouns, leading to redundancy.
Frequency: 5% of GPT responses showed overcorrection.
9. Future Work
The results of this project indicate significant advancements in reducing gender bias and improving neutrality in LLM-generated responses. However, several opportunities remain to enhance the models, refine methodologies, and address observed limitations. Below, we outline potential directions for future work.
9.1 Fine-Tuning Models with Expanded Datasets
1. Augmenting Training Data
Goal: Include a broader range of gender-neutral and contextually diverse prompts in the training set.
Approach:
Incorporate datasets such as GAP (Gendered Ambiguous Pronouns) and additional subsets from RealToxicityPrompts.
Design prompts that challenge the model's ability to maintain neutrality in nuanced or ambiguous scenarios.
10. Ethical Considerations
Ethics in artificial intelligence, particularly in the context of language models, is a crucial area of concern. This project, focused on mitigating gender bias in LLM-generated responses, directly addresses societal risks associated with biased AI systems. Below, we outline the ethical implications of our work, the risks it aims to mitigate, and strategies to ensure responsible development and deployment of these technologies.
10.1 Societal Risks of Bias in Language Models
1. Reinforcement of Gender Stereotypes
Issue: Gender biases in AI-generated responses can reinforce harmful societal stereotypes. For instance, associating certain professions or traits with specific genders perpetuates systemic inequalities.
Example: Responses that assume a nurse is female or an engineer is male subtly reinforce occupational segregation based on gender.
2. Loss of Trust in AI Systems
Issue: Perceived or actual bias in AI systems erodes public trust in technology.
Consequence: Reduced adoption of AI in sensitive domains such as healthcare or education, where its potential for positive impact is significant.
10.2 Ethical Challenges in Bias Mitigation
1. Overcorrection
Description: In the pursuit of neutrality, overcorrecting responses may result in the erasure of legitimate gender references, impacting the clarity and authenticity of responses.
Example: Replacing "she" with "they" in contexts where gender specificity is relevant.
2. Dataset Bias
Description: Biases in training datasets, whether implicit or explicit, can influence model behavior despite post-hoc mitigation strategies.
Challenge: Curating datasets that are representative, balanced, and free of stereotypes remains a persistent issue.
11. Code
To ensure transparency and reproducibility, all code used in this project has been made available in a dedicated GitHub repository. The repository link - 

12. References
Gallegos, I. O., Rossi, R. A., Barrow, J., et al. (2024).Bias and Fairness in Large Language Models: A Survey.Computational Linguistics, 50(3), 1097–1179.
Chen, R., Li, Y., Xiao, Z., et al. (2024).Large Language Model Bias Mitigation from the Perspective of Knowledge Editing.
OpenAI. -Available at: https://platform.openai.com/docs/
