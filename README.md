# AI Face Recognition Attendance System

An AI-powered attendance system using Face Detection and Face Recognition.

## Features
- Real-time face detection using YuNet
- Face recognition using MobileFaceNet
- Automatic attendance marking
- Add new students dynamically
- CSV attendance management

## Technologies Used
- Python
- OpenCV
- PyTorch
- NumPy
- Pandas

## Project Structure

``` id="6r6isj"
dataset/
main.py
YuNetFace.py
MobileFaceNet.py
create_embeddings.py
dataset_creation_yunet.py
students_marked.csv
```

## Run Project

```bash
python main.py
```

## Add New Student

```bash
python dataset_creation_yunet.py
```
