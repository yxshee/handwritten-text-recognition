
# 📜 Handwritten Text Recognition 



This repository implements a **Handwritten Text Recognition (HTR) model** using deep learning techniques. The goal is to convert handwritten text images into machine-readable text using a **Convolutional Recurrent Neural Network (CRNN)** with **CTC loss**.

✔️ Uses **CNNs** for feature extraction & **RNNs** for sequence learning  
✔️ Implements **CTC loss** for end-to-end training without explicit character segmentation  
✔️ Supports training, evaluation, and real-time inference  
✔️ Works with **IAM dataset** and custom datasets  

---

## **🛠️ Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yxshee/handwritten-text-recognition.git
cd handwritten-text-recognition
```
### **2️⃣ Install Dependencies**
Ensure you have **Python 3.7+** and run:
```bash
pip install -r requirements.txt
```

---

## **📂 Dataset Setup**
By default, the model is trained on the **IAM Handwriting Database**.  
### **1️⃣ Download IAM Dataset**
- Register at [Kaggle Dataset](https://www.kaggle.com/datasets/yashdogra/handwrittentext)
- Download and extract the dataset into the `data` folder

### **2️⃣ Organize Data Structure**
Ensure the dataset is structured as follows:
```
handwritten-text-recognition/
│── data/
│   ├── words/
│   ├── lines/
│   ├── ground_truth.txt
│── model/
│── src/
│── train.py
│── infer.py
```

---

## **🚀 Training the Model**
### **Run Training**
To train the model with the IAM dataset:
```bash
python train.py --epochs 50 --batch_size 32 --data_path ./data/
```
💡 Adjust hyperparameters as needed.

### **Monitor Training with TensorBoard**
```bash
tensorboard --logdir=logs/
```
Visit `http://localhost:6006/` to visualize training metrics.

---

## **📝 Running Inference**
### **Single Image Prediction**
To recognize text from an image:
```bash
python infer.py --image_path test_image.png
```
Expected output:
```
Recognized Text: "Hello, world!"
```

### **Batch Inference**
Run inference on a folder of images:
```bash
python infer.py --batch --image_folder ./test_images/
```

---

## **📊 Model Performance**
### **Evaluation**
To evaluate the trained model on a test set:
```bash
python evaluate.py --data_path ./data/
```
**Metrics:**  
✔️ Character Error Rate (CER)  
✔️ Word Error Rate (WER)  

---

## **🛠️ Custom Dataset Support**
To train on a **custom dataset**, format your data as follows:
```
data/
├── images/
│   ├── image1.png
│   ├── image2.png
├── labels.txt  # (format: "image_name text_label")
```
Then, train the model using:
```bash
python train.py --data_path ./custom_data/
```

---

## **📖 References & Acknowledgments**
- [IAM Handwriting Dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
- [CRNN Paper](https://arxiv.org/abs/1507.05717) - Convolutional Recurrent Neural Network
- [CTC Loss](https://distill.pub/2017/ctc/) - Connectionist Temporal Classification

---

## **🤝 Contributing**
We welcome contributions!  
📌 Feel free to submit **issues, pull requests, or feature suggestions**.

---

## **📜 License**
This project is **MIT licensed**. See the [LICENSE](LICENSE) file for details.

---

## **🌟 Connect & Support**
💬 For questions, open an **issue** or reach out via **GitHub Discussions**.  
⭐ If you find this project useful, **consider giving it a star!** 🌟

---







