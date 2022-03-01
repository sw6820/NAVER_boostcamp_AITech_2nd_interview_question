# NAVER_boostcamp_AITech_2nd_interview_question_AI
- Reference 이외의 질문들은 여러 면접시 직접 받은 중복되지 않은 질문들도 섞여있습니다. 

## AI
### [Deep Learning](https://github.com/sw6820/NAVER_boostcamp_AITech_2nd_interview_question/blob/main/AI/Topic/DL.md)
### [Machine Learning](https://github.com/sw6820/NAVER_boostcamp_AITech_2nd_interview_question/blob/main/AI/Topic/ML.md)
### [CV](https://github.com/sw6820/NAVER_boostcamp_AITech_2nd_interview_question/blob/main/AI/Topic/CV.md)
### [NLP](https://github.com/sw6820/NAVER_boostcamp_AITech_2nd_interview_question/blob/main/AI/Topic/NLP.md)
### [Reinforce Learning](https://github.com/sw6820/NAVER_boostcamp_AITech_2nd_interview_question/blob/main/AI/Topic/RL.md)
### [Recommender System](https://github.com/sw6820/NAVER_boostcamp_AITech_2nd_interview_question/blob/main/AI/Topic/RS.md)
### [MLOps](https://github.com/sw6820/NAVER_boostcamp_AITech_2nd_interview_question/blob/main/AI/Topic/MLOps.md)
### [Statistics / Math](https://github.com/sw6820/NAVER_boostcamp_AITech_2nd_interview_question/blob/main/AI/Topic/Statistics_Math.md)


### Deep Learning
- 딥러닝은 무엇인가요? 인공지능과 딥러닝과 머신러닝의 차이는?
- Cost Function과 Activation Function은 무엇인가요?
- Tensorflow, PyTorch 특징과 차이가 뭘까요?
- Data Normalization은 무엇이고 왜 필요한가요?
- 알고있는 Activation Function에 대해 알려주세요. (Sigmoid, ReLU, LeakyReLU, Tanh 등)
- 오버피팅일 경우 어떻게 대처해야 할까요?
- 하이퍼 파라미터는 무엇인가요?
- Weight Initialization 방법에 대해 말해주세요. 그리고 무엇을 많이 사용하나요?
- 볼츠만 머신은 무엇인가요?
- TF, PyTorch 등을 사용할 때 디버깅 노하우는?
- 뉴럴넷의 가장 큰 단점은 무엇인가? 이를 위해 나온 One-Shot Learning은 무엇인가?
- 요즘 Sigmoid 보다 ReLU를 많이 쓰는데 그 이유는?
- Non-Linearity라는 말의 의미와 그 필요성은?
- ReLU로 어떻게 곡선 함수를 근사하나?
    - ReLU의 문제점은?
- Bias는 왜 있는걸까?
- Bias vs Variance
    - Bias - Variance tradeoff는 무엇인가   
- Gradient Descent에 대해서 쉽게 설명한다면?
- 왜 꼭 Gradient를 써야 할까? 그 그래프에서 가로축과 세로축 각각은 무엇인가? 실제 상황에서는 그 그래프가 어떻게 그려질까?
- GD 중에 때때로 Loss가 증가하는 이유는?
- Back Propagation에 대해서 쉽게 설명 한다면?
- Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유는?
- GD가 Local Minima 문제를 피하는 방법은?
- 찾은 해가 Global Minimum인지 아닌지 알 수 있는 방법은?
- Training 세트와 Test 세트를 분리하는 이유는?
- Validation 세트가 따로 있는 이유는?
- Test 세트가 오염되었다는 말의 뜻은?
- Regularization이란 무엇인가?
    - ML 에서 언제 쓰나
    - Regularization vs Normalisation vs Standardization
- Batch Normalization의 효과는?
- Dropout의 효과는?
- BN 적용해서 학습 이후 실제 사용시에 주의할 점은? 코드로는?
- GAN에서 Generator 쪽에도 BN을 적용해도 될까?
- SGD, RMSprop, Adam에 대해서 아는대로 설명한다면?
- SGD에서 Stochastic의 의미는?
- 미니배치를 작게 할때의 장단점은?
- 모멘텀의 수식을 적어 본다면?
- 간단한 MNIST 분류기를 MLP+CPU 버전으로 numpy로 만든다면 몇줄일까?
- 어느 정도 돌아가는 녀석을 작성하기까지 몇시간 정도 걸릴까?
- Back Propagation은 몇줄인가?
- CNN으로 바꾼다면 얼마나 추가될까?
- 간단한 MNIST 분류기를 TF, PyTorch 등으로 작성하는데 몇시간이 필요한가?
- CNN이 아닌 MLP로 해도 잘 될까?
- 마지막 레이어 부분에 대해서 설명 한다면?
- 학습은 BCE loss로 하되 상황을 MSE loss로 보고 싶다면?
- 딥러닝할 때 GPU를 쓰면 좋은 이유는?
- GPU를 두개 다 쓰고 싶다. 방법은?
- 학습시 필요한 GPU 메모리는 어떻게 계산하는가?

### Machine Learning
- 알고 있는 metric에 대해 설명해주세요. (ex. RMSE, MAE, recall, precision ...)
    - Precision
    - Recall
    - F1 score
- 정규화를 왜 해야할까요? 정규화의 방법은 무엇이 있나요?
- Local Minima와 Global Minima에 대해 설명해주세요.
- 차원의 저주에 대해 설명해주세요.
- dimension reduction기법으로 보통 어떤 것들이 있나요?
- PCA는 차원 축소 기법이면서, 데이터 압축 기법이기도 하고, 노이즈 제거기법이기도 합니다. 왜 그런지 설명해주실 수 있나요?
    - PCA에서 components의 회전이 중요한 이유
- LSA, LDA, SVD 등의 약자들이 어떤 뜻이고 서로 어떤 관계를 가지는지 설명할 수 있나요?
- Markov Chain을 고등학생에게 설명하려면 어떤 방식이 제일 좋을까요?
- 텍스트 더미에서 주제를 추출해야 합니다. 어떤 방식으로 접근해 나가시겠나요?
- SVM은 왜 반대로 차원을 확장시키는 방식으로 동작할까요? SVM은 왜 좋을까요?
- 다른 좋은 머신 러닝 대비, 오래된 기법인 나이브 베이즈(naive bayes)의 장점을 옹호해보세요.
- 회귀 / 분류시 알맞은 metric은 무엇일까?
- Association Rule의 Support, Confidence, Lift에 대해 설명해주세요.
- 최적화 기법중 Newton’s Method와 Gradient Descent 방법에 대해 알고 있나요?
- 머신러닝(machine)적 접근방법과 통계(statistics)적 접근방법의 둘간에 차이에 대한 견해가 있나요?
- 인공신경망(deep learning이전의 전통적인)이 가지는 일반적인 문제점은 무엇일까요?
- 지금 나오고 있는 deep learning 계열의 혁신의 근간은 무엇이라고 생각하시나요?
- ROC 커브에 대해 설명해주실 수 있으신가요?
- 여러분이 서버를 100대 가지고 있습니다. 이때 인공신경망보다 Random Forest를 써야하는 이유는 뭘까요?
- K-means의 대표적 의미론적 단점은 무엇인가요? (계산량 많다는것 말고)
- L1, L2 정규화에 대해 설명해주세요.
- Cross Validation은 무엇이고 어떻게 해야하나요?
- XGBoost을 아시나요? 왜 이 모델이 캐글에서 유명할까요?
- 앙상블 방법엔 어떤 것들이 있나요?
- feature vector란 무엇일까요?tj
- 좋은 모델의 정의는 무엇일까요?
- 50개의 작은 의사결정 나무는 큰 의사결정 나무보다 괜찮을까요? 왜 그렇게 생각하나요?
- 스팸 필터에 로지스틱 리그레션을 많이 사용하는 이유는 무엇일까요?
- OLS(ordinary least squre) regression의 공식은 무엇인가요?
- ML 도입 이유
- 지도학습 vs 비지도학습 vs 강화학습
    - 지도학습
    - 비지도학습
    - 강화학습
- Bayes'Theorem?
- Naive bayes?
    - Naive bayes 에서 Naive는?
- PCA를 쓰는 경우
- SVM 알고리즘 자세히 설명하시오
    - SVM에서 support vectors가 무엇인가
- Cross-Validataion 이 무엇인가?
- ML에서 Bias는 무엇인가?
- Classification vs Regression
- F1 score가 뭔가? 어떻게 쓰나?
- Precision & Recall 정의
- Overfitting & Underfitting 대처법
- 인공 신경망이 무엇인가
- Loss Function & Cost function 정의, 차이점
- Ensemble Learning이 무엇인가?
- 어느 ML 알고리즘을 사용할지 어떻게 확신하나
- Outlier?
    - Outlier Values를 어떻게 다루나
- Random Forest 정의, 작동 방식
- K-means Clustering에서 K를 어떻게 고르나
- dataset의 Normality를 확인하는 방법
- Logistic Regression을 2가지 이상의 클래스에 사용 가능한가?
- Correlation vs Covariance vs Casuality
- Parametric & Non-Parametric 모델 설명
- 강화학습이 뭔가?
- Sigmoid vs Softmax 
- time series는 무엇인가요?
- box-cow transformation?
- random forest vs gradient boosting
- confusion matrix? 왜 
- marginalization?
    - marginalization 의 process
- data의 outliers를 다루는 방법
- 유명한 cross validation 기술들?
- Fixed Basis Function의 한계
- EDA technique
- boosting vs bagging
- ROC Curve
- Water Trapping Problem
- Decision tree 에서 hyper-parameter는?
- Cross-Validation의 역할?
- Pandas Profiling
- KNN에서 사용되는 거리 metric?
- pipeline이 무엇인가
- pruning의 이점
- the degree of freedom?
- Type I and Type II error
- utilities fraud detection에 관한 dataset이 있을때, 분류모델을 만들었고 그 모델 성능이 98.5%를 달성했을 때, 좋은 모델인가? 그렇다면 정당화하고, 아니라면 무엇을 할 수 있을까?
- 주어진 dataset에서 missing value나 corrupted value를 다루는 방법?
- How do you select important variables while working on a data set? 
- dataset이 주어졌을 때 사용할 ML 알고리즘을 고르는 방법?
- Hardware에 ML을 적용하는 방법 
- A data set is given to you and it has missing values which spread along 1 standard deviation from the mean. How much of the data would remain untouched?
- Stochastic Gradient Descent(SGD) vs Gradient Descent(GD)?
- 역전파 사용시 exploding gradient problem은?
- decision tree의 장단점


#### CV
- OpenCV 라이브러리만을 사용해서 이미지 뷰어(Crop, 흑백화, Zoom 등의 기능 포함)를 만들어주세요
- 딥러닝 발달 이전에 사물을 Detect할 때 자주 사용하던 방법은 무엇인가요?
- Fatser R-CNN의 장점과 단점은 무엇인가요?
- dlib은 무엇인가요?
- YOLO의 장점과 단점은 무엇인가요?
- 제일 좋아하는 Object Detection 알고리즘에 대해 설명하고 그 알고리즘의 장단점에 대해 알려주세요
    - 그 이후에 나온 더 좋은 알고리즘은 무엇인가요?
- Average Pooling과 Max Pooling의 차이점은?
- Deep한 네트워크가 좋은 것일까요? 언제까지 좋을까요?
- Residual Network는 왜 잘될까요? Ensemble과 관련되어 있을까요?
- CAM(Class Activation Map)은 무엇인가요?
- Localization은 무엇일까요?
- 자율주행 자동차의 원리는 무엇일까요?
- Semantic Segmentation은 무엇인가요?
- Visual Q&A는 무엇인가요?
- Image Captioning은 무엇인가요?
- Fully Connected Layer의 기능은 무엇인가요?
- Neural Style은 어떻게 진행될까요?
- CNN에 대해서 아는대로 얘기하라
- CNN이 MLP보다 좋은 이유는?
- 어떤 CNN의 파라미터 개수를 계산해 본다면?
- 주어진 CNN과 똑같은 MLP를 만들 수 있나?
- 풀링시에 만약 Max를 사용한다면 그 이유는?
- 시퀀스 데이터에 CNN을 적용하는 것이 가능할까?
- 이미지 처리에서 KNN을 사용하는 것이 가능한가?
- K-means vs KNN
- SVM 알고리즘에서 Kernel Trick?
- OOB error는 무엇이고 어떻게 발생하나?



#### NLP
<!-- - Word Representation
    - Bag-of-Word
    - TF-IDF
    - BM25
    - Word2Vec
    - GloVe
    - FastText
- Text Classification
    - Sentiment Analysis
    - Document Classification
    - Intent Classification
- Semantic/Syntatic Parsing
    - Named Entity Recognition
    - Semantic Role Labeling
    - Relation / Information Extraction
    - POS Tagging
    - Constituency Parsing
    - Dependency Parsing
- Disambiguation
    - Word-Sense Deambigution
    - Coreference Resolution
- Language Model
- Sequence Transduction
    - Machine Translation
    - Dialogue System
    - Summarization
    - Paraphrasing
- Advanced Attention mechanism
- Latent Random Variable Models
- Transfer Learning / Multi-task Learning
- Contextualized Word Representation (ELMo,CoVe) -->

- One Hot 인코딩에 대해 설명해주세요
    - one-hot encoding vs label encoding
    - 데이터셋의 차원에 어떻게 영향을 주나
- POS 태깅은 무엇인가요? 가장 간단하게 POS tagger를 만드는 방법은 무엇일까요?
- 문장에서 "Apple"이란 단어가 과일인지 회사인지 식별하는 모델을 어떻게 훈련시킬 수 있을까요?
- 뉴스 기사에 인용된 텍스트의 모든 항목을 어떻게 찾을까요?
- 음성 인식 시스템에서 생성된 텍스트를 자동으로 수정하는 시스템을 어떻게 구축할까요?
- 잠재론적, 의미론적 색인은 무엇이고 어떻게 적용할 수 있을까요?
- 영어 텍스트를 다른 언어로 번역할 시스템을 어떻게 구축해야 할까요?
- 뉴스 기사를 주제별로 자동 분류하는 시스템을 어떻게 구축할까요?
- Stop Words는 무엇일까요? 이것을 왜 제거해야 하나요?
- 영화 리뷰가 긍정적인지 부정적인지 예측하기 위해 모델을 어떻게 설계하시겠나요?
- TF-IDF 점수는 무엇이며 어떤 경우 유용한가요?
- 한국어에서 많이 사용되는 사전은 무엇인가요?
- Regular grammar는 무엇인가요? regular expression과 무슨 차이가 있나요?
- RNN에 대해 설명해주세요
- LSTM은 왜 유용한가요?
- Translate 과정 Flow에 대해 설명해주세요
- n-gram은 무엇일까요?
- PageRank 알고리즘은 어떻게 작동하나요?
- depedency parsing란 무엇인가요?
- Word2Vec의 원리는?
- 그 그림에서 왼쪽 파라메터들을 임베딩으로 쓰는 이유는?
- 그 그림에서 오른쪽 파라메터들의 의미는 무엇일까?
- 남자와 여자가 가까울까? 남자와 자동차가 가까울까?
- 번역을 Unsupervised로 할 수 있을까?

## Reference
- https://360digitmg.com/mlops-interview-questions-answers
- https://github.com/zzsza/Datascience-Interview-Questions?fbclid=IwAR1sVmRVTTRq73bwxHriYNyxDJG5mJzmtjQB01-jh16OefLFJCQsCyp7lf8
- https://github.com/boostcamp-ai-tech-4/ai-tech-interview
- https://www.mlstack.cafe/interview-questions/recommendation-systems
- https://www.mlstack.cafe/blog/recommendation-systems-interview-questions
- https://datascience.stackexchange.com/questions/tagged/recommender-system
- https://www.interviewbit.com/machine-learning-interview-questions/
- https://www.mygreatlearning.com/blog/machine-learning-interview-questions/
- https://www.springboard.com/blog/ai-machine-learning/machine-learning-interview-questions/
