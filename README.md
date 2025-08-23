# MIT-Designing-and-Building-AI-Products-and-Services

This repository contains my learning notes from MIT course on Designing and Building AI Products and Services. These
notes are self prepared to learn the course better and it is solely my work.

# Module 1

## Learn the basics of AI, ML, NLP and Deep learning:

[1] Machine learning:

- Teach machines to identify the patterns from the trained data.
- Allow computer to find the pattern so it can learn.
- 3 types of learning:
    - Supervised Learning
    - Unsupervised
    - Semi-Supervised.

[2] NLP:

- Allow computer to interact in human languages
- Ex: Language translation service; Chatbots.

[3] Deep Learning:

- Mimics the structure of Brain & function of Brain.
- Like brain has network of neurons to learn, reasoning ect. Deep learning has neural networks
- It has multiple layers of data processing between input and output.
- This allows to identify the very complex PATTERNS from the data like text, images, sound or combination of all.
- Ex: Computer vision - Face recognition; Self-driving cars ect.

## Introduction to AI Design Process:

### 4 stages of AI design process:

[1] Intelligence
[2] Business process
[3] AI Technology
[4] Tinkering

#### [1] first stage of the AI design process: Identifying the desired behavior of the AI product.

- Most important criteria of decisions is:
    - performance metrics : Error rate of image recognition.
    - scope : self-driving car. what all objects need to be recognized.

- Natural Language processing:
    - Sentiment Analysis
    - Summarize document
    - Summarize news
    - Answer questions
    - Maintain dialog
    - Analyze Reviews like google reviews

    - Sentiment analysis progress includes:
        - Coreference resolution. Ex: Identifying the expressions referring to text.
        - Word sense disambiguation
        - Parsing

#### [2] Business process:

- Two main areas where AI can help Business processes:
    - Strategic
    - Operational

[2.1] Strategic:

- Product Layer : Use the best AI product like build best fingerprint sensor.
- Customer Solution: Use AI product for best computer vision to incorporate in Self-driving car.
- Network Externalities : Build a large database of users so others can be more useful. Like more fingerprints in DB
  allows us to identify the people.

[2.2] Operational:

- Business process: Focus on operational process description like reduce the legal translation cost by using AI model in
  the legal translation service.
- Sensible Improvement targets: More measurable points like reduces the cost of translation by 80%.

#### [3] AI Technology

- Two main areas:
    - Intellectual Property (IP)
    - Data approach

[3.1] IP:

- Identify the process or business use case we are trying to achieve using AI technology.
    - Ex: Summarizing the claims documents using AI OR Sentimental analysis of sales calls transcripts.

[3.2] Data Approach:

- Data is the BASE for any AI technology.
- Metadata (which is data of the data) includes - Date, time, Collection mechanism and Content description.
- Thousands of patents are awarded yearly for a wide spectrum of AI use cases.
- Any AI use case should justify below 3 areas:
    - Network externalities
        - larger community size
        - larger asset attractiveness
        - bigger value per user

    - System lock-in
        - Customer lock-in
        - More revenue per customer
        - Bigger loyalty

    - Economies of scale
        - Bigger R&D budget
        - Bigger margins
        - Low operating cost

[3.3] Exercise:

1. News Article: I read the news article about how AI is reshaping the auto-insurance industry.

https://www.kiro7.com/news/role-ai-shaping-auto-insurance/OPJ5GK7ILFJQJLCRBARDOD5MEU/#:~:text=Can%20Smart%20Technology%20Help%20You,pricing%20system%20feel%20fairer%20overall.

2. Advantages of AI : AI analyses users driving pattern like - speed, braking, amount of driving and majority what time
   of the day user drives. These data identifies tha pattern of driving and co-relation with safe driving and accidents.

[2.1] Network externalities:

When more users joined this program, machine learning models will get vast amount data to identify the patterns and find
correlation.
AI can differentiate between low risk and high risk drivers. This provides lower premium to safe drivers and thus it
gives large audience attractiveness.
AI can find the pool of safe drivers, this allows lower premium cost to safe drivers.

[2.2] System lock-in

AI establishes the pattern of safe driving, this allows insurance companies to provides discount and low cost insurance
which boost customer lock-in.
Lower premium attracts more customers which provides more revenue. Customer stays longer and creates loyalty with the
business.

[2.3] Economies at scale:

More revenue allows the auto-insurance to spend more on R&D to improve the AI models to make it more efficient.
The safer drivers have low probability of claims and thus lower money spend on them, This increases profit margins.

AI models identifies the safe and risky drivers by analyzing their driving pattern. This allows auto-insurance company
to have less human resources in reviewing the application and claims which provides low operating cost.

[3] Business Perspective:

AI allows the business to identify and segregate the safe and risky drivers based on their driving profile. This allows
business to provides more discount and fair and competitive price to the customer which attracts more customer and
generates the loyalty for the long term commitment.

AI analysis allows the business to provide the fair prices to customer based on their driving history which other
competition might not have.

[4] Customer perpective:

- Customer understands that AI is gathering the data to analyze the driving profile. This encourages customers to follow
  specific rules to keep getting the discounted price for the auto-insurance. This enables the customers to get the
  lower premium.

#### [4] Tinkering

[4.1] Software development

[4.2] AI Cancers:

- Adversarial attacks:
    - Any adversarial noise in the data can create the wrong output.
    - Ex: Small stickers on stop sign can cause Self driving cars believe that stop sign is actually speed post which
      might cause issue.

- Lack of generalization:
    - Translation accuracy from language A to B does not guarantee translation accuracy of language A to C.
    - For ex: Face recognition softwares developed by Microsoft Azure OR aws has millions of face images which
      over-trained model on images but did not consider the same image with makeup and other edits.

- Bias:
    - GPT2 had a biased responses.

- Explainability:
    - Wolf vs dogs image recognition shows biased result because wolf with snow in background and dog with home
      background.

- Unintended Behavior:
    - Ex: Robot with AI fell down on escalator in a mall and injured other customer.
    - NLP model gave bad advice to the customer.

#### [5] Activity on Software development plan cost analysis:

[1] Will the costs for integrating your virtual assistant device with an NLP align with the development costs itemized
in the spreadsheet?

- No, NLP integration with the virtual assistant is very complex project. In the cost analysis worksheet, we did NOT
  consider the cost associate with the model fitting and accuracy testing.
- We need to allocate the budget and resources to perform comparison analysis of the different NLP models.
- Also need to verify the accuracy of the various models to determine which NLP models is the right solution to our
  problem.

[2] In which areas can you reduce your development costs?

- I would save cost on Hardware and Network cost by going with Open AI models and API integration. This way, we do NOT
  need to provision heavy processing GPUs and rather make the API calls to the Open AI NLP models from cloud.
- I'd also outsourced some of the development work related to integration to offshore countries to reduce the cost of
  software development.

[3] In which areas will your development costs exceed the estimates presented in the spreadsheet?

- Consider the complex nature of the NLP model integration, the labor costs might increase. There are certain unknown
  blockers the development team might face which increases the estimated time of development and thus it increases the
  cost.
- Another area could be the maintenance, when the virtual assistant gets more traction with user and increases the
  usage, it significantly consumes more GPUs utilization which end-up increasing the cost.

[4] Does the spreadsheet account for the costs of your data needs?

- The speadsheet mentioned about the software licensing cost but did NOT mention anything about the NLP model
  integration cost. It should be clearly defined about the type to NLP model can be used, comparison of various models
  to identify the optimal model and scalability of NLP model when the user base growth. Also, it did not mention the
  token optimization to avoid paying higher for each query.

[5] What is your contingency plan if you exceed your resources or allotted project development time?

- In cost analysis, we put 10% of total implementation cost as a contingency. Based on the experience and research I
  feel the contingency plan should be at least 20 - 30% of the total allocated budget.
- I'd divide the project into iteratively development plan. I should analyse all the features and prioritize the most
  useful ones. Also, I finalize the deliverables for the Minimum Viable Product (MVP) and based on the remaining budget,
  I'd go for the additional features.

[6] Is this plan worth funding? Why or why not?

- Though this project performed quite analysis and prepared project development documentation, It misses out many
  important areas like model comparison and complex development cost. Also it takes very optimistic approach on the
  timeline, which considering the complex nature of the development plan is very tightly set. In addition to that,
  contingency plan is also very optimistic and did not provide enough margins for the error. I feel the plan is not
  worth the funding at this point and deep re-evaluation is necessary.
