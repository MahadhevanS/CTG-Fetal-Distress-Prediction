# Team Guide: Running Your Models on Google Colab 🚀

Hey team! To make sure our models train fast and our benchmarking is 100% fair, we are training **all models** on Google Colab using free GPUs.

Here is a super simple, 5-minute guide on how to run your code on Colab.

---

### Step 1: Open Colab and Turn on the GPU
1. Go to [colab.research.google.com](https://colab.research.google.com/) and click **New Notebook**.
2. In the top menu, click **Runtime** $\rightarrow$ **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** and click Save.

### Step 2: Connect to the Shared Google Drive
We have already processed the CTG data into PyTorch `.pt` files. They are saved in a Google Drive folder. You just need to connect your Colab notebook to Drive.

Paste this code into the first cell and click the "Play" button to run it:
```python
from google.colab import drive
drive.mount('/content/drive')
```
*(A popup will ask for permission to access your Google Drive. Click Allow.)*

### Step 3: Clone Your Code from GitHub
You will write your model code on your laptop and push it to your specific GitHub branch (e.g., `models_set_1`). Colab just downloads your branch and runs it!

Paste this into the next cell, change `<your-repo-url>` and `<your-branch-name>`, and run it:
```bash
!git clone <your-repo-url>
%cd CTG
!git checkout <your-branch-name>
!pip install -r requirements.txt
```
*(This downloads your code into the Colab environment and installs PyTorch, etc.)*

### Step 4: Train Your Model!
Now you just run the Python script you wrote to train your model. Make sure to point the script to where the data lives on Google Drive, and make sure your script saves the `.pth` model weights back to Drive so you don't lose them!

Paste this into the final cell and run it:
```bash
!python src/models/train_YOUR_MODEL.py --data_dir /content/drive/MyDrive/CTG_Project/data/processed/ --save_dir /content/drive/MyDrive/CTG_Project/models/
```
*(Replace `MyDrive/CTG_Project/` with wherever the team lead saved the shared folder).*

---

### 💡 Quick Tips:
- **Write code locally, run on Colab**: Don't write 500 lines of code in Colab. Write it in VSCode, push to your branch, and use Colab just for the GPU power.
- **Don't touch the data**: The dataset is completely frozen. If your model throws a shape error (e.g., expecting `[Batch, 4800, 2]` instead of `[Batch, 2, 4800]`), **fix your model's code**, do not change the dataset!
- **Log your results**: Once your model finishes training, copy the final AUROC and F1 scores into the `docs/model_inferences_log.md` file and push it to GitHub!
