# ssm-cs221
## 1. Introduction
In recent years, the application of artificial intelligence (AI) has seen rapid growth, transforming a variety of industries from healthcare to education, recruitment, and beyond. However, as AI systems such as Large Language Models (LLMs) become more prevalent, they also inherit biases from the data on which they are trained. This phenomenon is not only an ethical concern but can also have real-world implications in decision-making processes, especially when biases align with societal stereotypes.
Gender bias, in particular, has been a subject of extensive discussion due to its pervasiveness in many domains. The language produced by AI models often reflects deeply ingrained societal norms and historical data, perpetuating stereotypes related to gender roles. For example, models may associate certain professions or tasks with a specific gender, such as "nurse" being more frequently paired with female pronouns, and "engineer" with male pronouns. This type of bias in AI responses can influence how AI systems interact with users and can have far-reaching consequences, from reinforcing gender stereotypes in educational materials to influencing hiring decisions, thereby contributing to gender inequality.
The objective of this study is to investigate gender bias in LLMs, specifically through the analysis of the Mistral-7B model, and to propose techniques for mitigating such bias. This study combines probabilistic modeling with advanced natural language techniques in order to dynamically identify and correct biased outputs in real-time, significantly improving gender neutrality in the model's responses.

## 2. Dataset
The dataset forms the cornerstone of our analysis, as it provides the contextual diversity required to evaluate and mitigate gender bias in large language models (LLMs). This section provides a detailed exploration of the dataset used, its characteristics, preprocessing steps, and relevance to the objectives of this project.
2.1 Dataset Source
The primary dataset used in this study is the WinoBias/WinoGender dataset, a well-known benchmark for evaluating gender bias in NLP models. The dataset comprises sentences with ambiguous pronoun references and varying contexts, focusing on occupational and societal roles. Each example includes:
Tokens: The words that make up the sentence.
Coreference Clusters: Pre-annotated clusters indicating the intended referent for each pronoun.
2.2 Dataset Characteristics
The final dataset used for the study consisted of:
790 Rows: Each row represents a sentence with ambiguous pronoun references.
Occupational Roles: Examples such as "The engineer fixed the machine quickly," which test for gender association with traditionally male or female roles.
Neutral Prompts: Examples that deliberately avoid specifying gender to test the model's baseline assumptions.
Balanced Representation: Equal distribution of prompts involving male, female, and neutral contexts to ensure fairness in evaluation.

## 3. Baseline
The baseline forms the foundation for evaluating the progress and improvements achieved through the main approach. It establishes the initial performance metrics and highlights the areas where the model may require refinement to address gender bias effectively. This section elaborates on the baseline setup, methodology, and findings.
3.1 Baseline Model
The baseline evaluation was conducted using the Mistral-7B-v0.1 model, accessed via the Hugging Face Inference API. Mistral-7B is a state-of-the-art language model designed for general-purpose natural language processing tasks. It was chosen due to its competitive performance on several NLP benchmarks and its capability to handle nuanced tasks such as coreference resolution.
3.2 Baseline Implementation
To evaluate the model’s performance on gender bias and alignment tasks, the following methodology was employed:
Prompt Construction: Sentences from the dataset were formatted into prompts designed to test the model’s ability to resolve ambiguous pronouns. For example:
Prompt: Please resolve the coreferences in the following sentence and specify which noun each pronoun refers to. Sentence: "The mechanic greets with the receptionist because she was in a good mood."
Coreference Resolution: The model was tasked with identifying the antecedent of each pronoun in the sentence. The output was analyzed to determine whether the pronoun resolution aligned with the dataset’s annotations.
Pronoun Analysis: The frequency of gendered pronouns ("he," "she," "they") was computed to identify any inherent biases in the model’s responses.
Sentiment Analysis: Each response was analyzed using the TextBlob library to classify the sentiment as positive, negative, or neutral.
Stereotype Score: A stereotype score was calculated based on the association of specific professions with gendered pronouns. For example:
"Engineer" with "he" or "Nurse" with "she" would increment the stereotype score.
3.3 Baseline Metrics
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
## 4. Main Approach
The primary objective of this project is to address the limitations identified in the baseline by developing a dynamic and adaptive system for resolving gender bias in model outputs. To achieve this, we integrate a State Space Model (SSM), specifically a Hidden Markov Model (HMM), into the analysis pipeline. This section describes the methodology, design, and implementation of our main approach.
4.1 Overview of the Main Approach
The main approach integrates an HMM with the existing Mistral-7B language model to:
Dynamically assess the bias state of model-generated outputs.
Transition between bias states (e.g., neutral → biased masculine) during text generation.
Adjust outputs in real-time to favor neutrality while preserving semantic consistency.
The HMM serves as a probabilistic framework to model sequential transitions between bias states. This allows for a systematic adjustment of outputs by leveraging state transition probabilities derived from linguistic and contextual features.
4.2 Hidden Markov Model (HMM) Framework
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
## 5. Evaluation Metric
Evaluation metrics are critical in quantifying the success and limitations of our approach. For this project, we employed a combination of quantitative and qualitative metrics to evaluate the performance of the baseline model, the GPT-adjusted responses, and the HMM-adjusted responses. The metrics were designed to measure both gender bias and context relevance while ensuring that semantic meaning was preserved.
5.1 Metrics Overview
5.1.1 Gender Bias Score (GBS):
This metric quantifies the extent of gender bias in a response based on pronoun usage:
GBS=Count(Female Pronouns)−Count(Male Pronouns)GBS = \text{Count(Female Pronouns)} - \text{Count(Male Pronouns)}GBS=Count(Female Pronouns)−Count(Male Pronouns)
Positive GBS: Indicates a female bias.
Negative GBS: Indicates a male bias.
Zero GBS: Indicates gender neutrality.
5.1.2 Context Alignment Score (CAS):
This binary metric evaluates whether the response aligns with the expected gender context of the prompt:
CAS={1,if response matches expected gender context0,otherwiseCAS = \begin{cases} 1, & \text{if response matches expected gender context} \\ 0, & \text{otherwise} \end{cases}CAS={1,0,​if response matches expected gender contextotherwise​
5.1.3 Neutrality Score (NS):
A numerical score from 1 to 5, where:
1=Very biased1 = \text{Very biased}1=Very biased
5=Completely neutral5 = \text{Completely neutral}5=Completely neutral
Neutrality scores were obtained using the GPT-3.5-turbo model by prompting it to evaluate each response's neutrality.
5.1.4 Pronoun Distribution:
This metric tracks the frequency of pronouns ("he," "she," "they") in the model's output. It helps to identify imbalances and biases in pronoun usage across responses.
5.1.5 Stereotype Score:
This score measures alignment with occupational gender stereotypes:
Stereotype Score=∑i=1nMatches(Rolei,Pronouni)\text{Stereotype Score} = \sum_{i=1}^n \text{Matches}(\text{Role}_i, \text{Pronoun}_i)Stereotype Score=i=1∑n​Matches(Rolei​,Pronouni​)
5.2 Quantitative Evaluation
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
## 6. Results & Analysis
The results and analysis section focuses on the comparative evaluation of the baseline model, the HMM-adjusted responses, and GPT-adjusted responses. This section highlights the quantitative and qualitative findings and provides insights into the effectiveness of each approach. Detailed examples and visualizations are included to illustrate the performance trends and the impact of the adjustments on gender bias and neutrality.
6.1 Results
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

