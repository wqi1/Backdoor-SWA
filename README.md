# Static Weight Analysis Guided Trigger Reverse Engineering for DNN Backdoor Detection

This is a demo for the paper: Static Weight Analysis Guided Trigger Reverse Engineering for DNN Backdoor Detection

The code is tested on TrojAI datasets (round2-3). The datasets can be accessed at [TrojAI](https://pages.nist.gov/trojai/docs/data.html)


### Dependences
Python 3.6, torch>=1.5.1, torchvision>=0.6.1, numpy

### File Description
The file `reverse_polygon.py` is for detecting models trojaned by polygon patch triggers, and `reverse_filter.py` is for 
detecting models trojaned by Instagram Filter triggers. 

### Quick Start
To run the code, please first modify the dataset path and output path,
then simply run the command:
```bash
python reverse_polygon.py
```
or
```bash
python reverse_filter.py
```
The code will output a `test.csv` recording the target label identification results by static weight analysis,
the running information for each epoch, and final model-level prediction results.
