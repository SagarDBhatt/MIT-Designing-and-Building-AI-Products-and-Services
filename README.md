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

#### [6] AI Cancers Activity:

[1] News Article: AI can perpetuate racial bias in insurance
underwriting (https://finance.yahoo.com/news/ai-perpetuates-bias-insurance-132122338.html)

[2] Describe how your selected product or process integrates AI.

- Recently, more and more Auto insurance companies rely on AI models for the insurance underwriting and predicting
  drivers profile.
- These models consider the parameters like acceleration, brakes, amount of driving and more. In addition to that AI
  models also considers the zip codes where the drivers drive and frequency of driving to the particular area. It also
  identifies the probabilities of accidents in certain areas based on the past accident data to train the AI model on
  the process of insurance underwriting.

[3] Identify the type of AI cancer that the article describes.

- Article mentioned that AI model uses zip codes as a mask to identify the socioeconomic status of the person. AI give
  biased results based on the zipcode data which passively profile the socioeconomic structure of the person.
- This bias can be seen in terms of racially profiling the driver and predicting the underwriting of the auto insurance
  premium.
- This includes the principles of lack of generalization and biased behaviors.

[4] Discuss how you’d prevent or mitigate the effects of AI cancer in your selected product or process.

- I'd incorporate the more generalized data and parameters while training the models. This will allow the AI model to
  avoid any biased against the race or socioeconomic factors. Also, I use un-masked parameters which nullify the proxy
  behavior in training the model.

# Module 2: Artificial Intelligence Technology Fundamentals: Machine Learning.

#### In this module, you'll learn how to discriminate between different machine learning algorithms. You will learn about each topic from theoretical as well as practical points of view. You'll begin with a discussion of linear classifiers and decision trees and then look at machine learning algorithms that use probabilistic approaches, such as Bayesian or regression models.

#### [2.1] Machine Learning:

- ML is a field of AI which allows computers to learn from data without being explicitly programmed. In other words,
  computer identifies the patterns based on the data provided, based on this pattern machine can derive the results.

- There are 3 types of ML:
    - Supervised
    - Semi-Supervised
    - Unsupervised.

#### [2.1.1] Supervised ML

- In Supervised learning, ML models are trained on LABELED data. Labels are desired output of the data.
- For Ex: ML models to identify spam OR authentic emails. Here, ML model is trained on datasets of emails which has
  labeled emails into spam and authentic emails.
- This helps models to analyze the pattern between spam and authentic emails. This will allow ML model to identify spam
  emails based on pattern analysis from training labeled datasets.

[A] Regression :

- Common Algorithms:
  [1] Linear Regression : Creates best fit line to predict a value based on single input variables.Ex: Predicting the
  house price based on num of bedrooms, predicting stock price based on gdp data ext.
  [2] Support Vector Regression: For more non-linear relationships.

[B] Classification : This is used for predicting discrete categories, such as classifying an image as a cat or a dog.

- Common algorithms include:
  [1] Logistic regression: This predicts the probability of an event belonging to a specific class (often used for
  binary classification).
  [2] Decision trees: This creates a tree-like structure to classify data based on a series of questions about the
  features.
  [3] k-nearest neighbors (KNN): This classifies data points based on the labels of their nearest neighbors in the
  training data.

[ ] Taxonomy of Machine Learning:

![img.png](img.png)

![img_1.png](img_1.png)

# Module 3 : Artificial Intelligence Technology Fundamentals: Deep Learning

- Deep learning provides greater flexibility and power in building AI products. You'll begin the module with a
  discussion of artificial neural networks and learn about its basic component: the neuron.
- Next, you’ll explore the multi-layer perceptron — an advanced structure consisting of multiple layers of artificial
  neural networks. You’ll also consider architectures that can be used to customize neural networks, such as
  autoencoders, and explore various types of neural networks and their applications in the form of convolutional neural
  networks and more sophisticated deep neural networks.
- Finally, you’ll reach the end of the module with an investigation into an architecture developed to deal with sequence
  data: recurrent neural networks.

### [3.1] Task - Identify the application of Neural Network OR deep learning algorithm in business.

- Neural Network / Deep learning Algorithm in Auto Insurance claims
  appraisal - https://scholar.smu.edu/cgi/viewcontent.cgi?article=1181&context=datasciencereviewLinks to an external
  site.

- Neural networks and deep learning algorithms, particularly Convolutional Neural Networks (CNNs), are applicable in the
  auto insurance industry for tasks such as claims processing, damage assessment, fraud detection, and risk modeling. A
  paper in the SMU Data Science Review explores applying deep learning, computer vision, and neural networks to
  automotive
  damage appraisal, proposing a workflow combining image analysis and statistical modeling to predict claim costs.

- The article indicates that this method, using an advanced neural network algorithm and techniques like Mask R-CNN, can
  improve accuracy and efficiency, potentially cutting labor costs by half and reducing appraisal time from days to
  hours.

### [3.2] An artificial neural network (ANN) is a computer program inspired by the workings of the human brain.

- As the brain uses interconnected neurons, an ANN uses artificial neurons connected in a network. Each artificial
  neuron receives input, such as a pixel value (like handwritten numbers), and processes it to produce an output.
- This output is passed to other neurons in the network. The strength of the connection between neurons, called weights,
  determines the influence one neuron has on another. During training, the network adjusts these weights based on the
  examples it receives so that it gets better at recognizing patterns in the data.

![img_2.png](img_2.png)

#### Basic neural network architecture with input, hidden, and output layers.

- The basic parts of an ANN are:

- Input layer: This is the first layer of the network, where the input data is fed into the network. Each neuron in this
  layer represents a feature or input variable.

- Hidden layers: These are layers in between the input and output layers. Each hidden layer consists of neurons that
  perform computations on the input data. The number of hidden layers and neurons in each layer can vary depending on
  the
  complexity of the problem.

- Weights: Each connection between neurons in adjacent layers has a weight associated with it. These weights determine
  the
  strength of the connection and are adjusted during the training process to improve the network's performance.

- Activation function: Each neuron in the network applies an activation function to the weighted sum of its inputs. This
  function introduces nonlinearity to the network, allowing it to learn complex patterns in the data.

- Output layer: This is the final layer of the network, where the network's output is generated. The number of neurons
  in
  the output layer depends on the type of problem the network is solving (e.g., classification or regression).

- Bias: Each neuron typically has an associated bias, which allows the network to learn the optimal decision boundaries
  for the data.

- Loss function: This function measures how well the network is performing by comparing its output to the true labels in
  the training data. The goal of training is to minimize this loss function.

- Optimizer: The optimizer is responsible for adjusting the weights of the network during training to minimize the loss
  function. Popular optimizers include gradient descent, stochastic gradient descent (SGD), and Adam, which are used to
  update the model's parameters based on the gradients calculated during backpropagation.

- Layers connectivity: The way neurons in adjacent layers are connected can vary. In a fully connected layer, each
  neuron is connected to every neuron in the adjacent layer.

Perceptron model: inputs, weights, summation, activation, output.

![img_3.png](img_3.png)

Comparison Between Biological Neurons and Artificial Neurons

ANNs are inspired by the structure and function of the human brain.

Structure: Both human neurons and artificial neurons have a similar structure, with inputs, a processing unit, and
outputs.
Function: Information is passed from one neuron to another through connections.
Learning: Both systems are capable of learning from experience. In the human brain, this involves synaptic plasticity,
where the strength of connections between neurons is adjusted based on the patterns of activity. In ANNs, learning is
typically achieved through algorithms that adjust the weights of connections between neurons based on the error in the
network's predictions.

Activation: Both human neurons and artificial neurons use an activation function to determine their output based on the
inputs they receive. This allows both systems to model complex, nonlinear relationships in data.

Parallel processing: Both systems are capable of parallel processing, where multiple neurons or processing units can
perform computations simultaneously.

Fault tolerance: Both systems exhibit some degree of fault tolerance. In the human brain, this is due to the redundancy
of connections between neurons. In ANNs, this can be achieved through techniques such as dropout, which randomly removes
connections between neurons during training to prevent overfitting.

#### Step-by-Step Learning Using Gradient Descent

![img_4.png](img_4.png)

Neural network diagram illustrating forward and backward propagation processes.

Initialize weights: Start by initializing the weights of the neural network to small random values. These weights are
the parameters that the network will learn during training.
Forward pass: Perform a forward pass through the network. This involves propagating the input data through the network
to compute the output. Each neuron in the network calculates its output based on the weighted sum of its inputs and
applies an activation function to this sum.
Compute loss: Calculate the loss, which is a measure of how well the network's output matches the true output (the
labels in the training data).
Compute gradient of loss: Calculate the gradient of the loss function with respect to the weights of the network.
Update weights: Update the weights of the network using the gradient descent algorithm. The weights are updated in the
opposite direction of the gradient to minimize the loss.
Repeat: Repeat steps 2–5 for a fixed number of iterations (epochs) or until the loss converges to a satisfactory level.
Each iteration through the entire dataset is called an epoch.
Evaluate: After training is complete, evaluate the performance of the network on a separate validation or test dataset
to see how well it generalizes to new, unseen data.
Adjust hyperparameters: Experiment with different hyperparameters (learning rate, number of hidden layers, number of
neurons per layer, etc.) to improve the performance of the network.
By iteratively updating the weights of the network using the gradient descent algorithm, the network learns to make
better predictions and minimize the loss function, thereby improving its performance on the task it is trained for.

#### Types of Activation Function Used in ANNs

There are different types of activation functions used in ANNs. Some of the common ones are the following:

Sigmoid: The sigmoid function is a smooth, S-shaped curve that maps input values to a range between 0 and 1. It is often
used in the output layer of a binary classification problem, where the network needs to predict probabilities.
Hyperbolic tangent (tanh): The tanh function is similar to the sigmoid function but maps input values to a range between
−1 and 1. It is often used in hidden layers of the network.
Rectified linear unit (ReLU): The ReLU function is a simple nonlinear function that returns 0 for negative inputs and
the input value for positive inputs.
Leaky ReLU: Leaky ReLU is similar to ReLU but allows a small, nonzero gradient for negative inputs. This helps to
mitigate the "dying ReLU" problem, where neurons can become inactive and stop learning.
Softmax: The softmax function is often used in the output layer of a multi-class classification problem, where the
network needs to predict probabilities for each class. It converts the raw output values into probabilities that sum up
to 1.

![img_5.png](img_5.png)

Other terminologies related to deep learning:

Epoch: One complete presentation of the entire training dataset to the learning algorithm — multiple epochs are often
required to adequately train a model

Batch size: The number of training examples utilized in one iteration of model training

Dropout: A regularization technique that involves randomly setting a fraction of input units to 0 at each update during
training time, which helps to prevent overfitting

Transfer learning: Leveraging a pretrained model on a new, related problem — popular in deep learning where large
datasets are required to train a model from scratch
