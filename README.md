<div align="center">
  <h1>Multi-Modal Music Genre Classification</h1>
  <p>
    An exploration of NLP and Computer Vision techniques to classify music genres from symbolic music data (MXL) and sheet music images (PDF).
  </p>
</div>

<hr>

<h3>
  <a href="#-project-overview">Project Overview</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#-key-skills--technologies">Skills & Technologies</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#-repository-structure">Repository Structure</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#-getting-started">Getting Started</a>
</h3>

<hr>

<h2>📌 Project Overview</h2>

<p>
  This project tackles the challenge of music genre classification by applying machine learning models to two distinct data modalities. The primary goal is to demonstrate a versatile skill set in handling, processing, and modeling different types of data—symbolic and visual—to solve a single, cohesive problem.
</p>
<p>
  The "why" behind this project is to showcase an end-to-end data science workflow, from rigorous exploratory data analysis (EDA) and preprocessing of a complex, imbalanced dataset to the implementation and evaluation of state-of-the-art deep learning models. It serves as a practical demonstration of my ability to approach a problem from multiple angles and leverage the appropriate tools for each task.
</p>

<h3>🎵 Part 1: NLP Approach — Classification from Symbolic Music Data (MXL)</h3>
<p>
  This component treats music as a language. By parsing the symbolic notation in MusicXML (<code>.mxl</code>) files, we can apply powerful NLP models to learn and classify genres.
</p>
<ul>
    <li>
        <strong>Data Source</strong>: The project utilizes the <strong>Pop Music Dataset with Mxl (PDMX)</strong>, which contains over 250,000 songs.
    </li>
    <li>
        <strong>Exploratory Data Analysis (EDA)</strong>: A thorough EDA was conducted (see <code>nlp/EDA.ipynb</code>) to understand the dataset's characteristics. Key steps included:
        <ul>
            <li>Handling a significant number of unlabeled instances (171k out of 254k).</li>
            <li>Consolidating 20+ genres into a more manageable set of 8 distinct classes.</li>
            <li>Addressing severe class imbalance using <strong>Random Undersampling</strong> to create a final, balanced dataset of 16,118 instances, preventing model bias towards the majority class ("Classical").</li>
        </ul>
    </li>
    <li>
        <strong>Modeling</strong>: Transformer-based architectures like <strong>BERT</strong> and <strong>DistilBERT</strong> are employed to classify the tokenized musical sequences.
    </li>
</ul>

<h3>🖼️ Part 2: Computer Vision Approach — Classification from Sheet Music Images</h3>
<p>
  This component treats music as a visual medium. Sheet music pages, derived from PDF files, are converted into images and fed into Convolutional Neural Networks (CNNs) to visually classify the genre.
</p>
<ul>
    <li>
        <strong>Data Source</strong>: The same PDMX dataset is used, leveraging the PDF files associated with each song.
    </li>
    <li>
        <strong>Preprocessing</strong>: PDF files are converted into images suitable for input into CNN models. Data augmentation techniques are used to improve model generalization.
    </li>
    <li>
        <strong>Modeling</strong>: This approach leverages CNNs for image-based genre classification. A supplementary report included in this project (<a href="https://github.com/s2751148/mlp_cw2/blob/main/s2751148_report.pdf" target="_blank">s2751148_report.pdf</a>) provides an in-depth analysis of training deep CNNs (VGG38), diagnosing the <strong>vanishing gradient problem</strong>, and implementing solutions like <strong>Batch Normalization</strong> and <strong>Residual Connections</strong>. This demonstrates a deep theoretical and practical understanding of modern computer vision architectures.
    </li>
</ul>


<h2>🛠️ Key Skills & Technologies</h2>
<p>This project showcases proficiency in the following areas:</p>
<ul>
    <li><strong>Data Science & Machine Learning</strong>:
        <ul>
            <li>Exploratory Data Analysis (Pandas, Matplotlib, Seaborn)</li>
            <li>Advanced Data Preprocessing & Cleaning</li>
            <li>Handling Class Imbalance (Undersampling)</li>
            <li>Multi-Modal Feature Engineering & Modeling</li>
        </ul>
    </li>
    <li><strong>Natural Language Processing (NLP)</strong>:
        <ul>
            <li>Transformer Architectures (BERT, DistilBERT)</li>
            <li>Custom Tokenization for Symbolic Data</li>
            <li>Sequence Classification</li>
        </ul>
    </li>
    <li><strong>Computer Vision (CV)</strong>:
        <ul>
            <li>Convolutional Neural Networks (CNNs, VGG)</li>
            <li>Image Classification</li>
            <li>Understanding of Deep Learning Challenges (Vanishing Gradients)</li>
            <li>Architectural Enhancements (Batch Normalization, Residual Connections)</li>
        </ul>
    </li>
    <li><strong>Programming & Tools</strong>:
        <ul>
            <li><strong>Languages</strong>: Python</li>
            <li><strong>Libraries</strong>: PyTorch/TensorFlow, Scikit-learn, Pandas, NumPy</li>
            <li><strong>Development</strong>: Jupyter Notebooks, Git, GitHub</li>
        </ul>
    </li>
</ul>

<h2>📁 Repository Structure</h2>
<p>The repository is organized to separate the two main approaches while maintaining a shared data source.</p>
<pre><code>
.
├── 📂 computer_vision
│   ├── 📄 ImageClassGenre.ipynb
│   └── 📄 cv-notebook.ipynb
├── 📂 data
│   ├── 📄 dataset.csv
│   ├── 📄 preprocessed_dataset_json.zip
│   └── 📄 vocab.csv
├── 📂 nlp
│   ├── 📄 BERT_classifier.ipynb
│   ├── 📄 distilbert&test.ipynb
│   ├── 📄 mxlClassGenre.ipynb
│   ├── 📄 mxl_to_musicxml.ipynb
│   ├── 📄 mxl_tokenizer.py
│   ├── 📄 transformer_model.ipynb
│   └── 📄 EDA.ipynb
├── 📄 s2751148_report.pdf
└── 📄 README.md
</code></pre>

<h2>🚀 Getting Started</h2>

<h3>Prerequisites</h3>
<p>Ensure you have Python 3.8+ and pip installed. This project uses standard data science libraries like PyTorch/TensorFlow, Pandas, and Scikit-learn.</p>

<h3>Installation & Usage</h3>
<ol>
    <li><strong>Clone the repository:</strong>
        <pre><code>git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name</code></pre>
    </li>
    <li><strong>Install dependencies:</strong>
        <pre><code>pip install -r requirements.txt</code></pre>
        <em>(Note: A <code>requirements.txt</code> file would need to be created for this step).</em>
    </li>
    <li><strong>Explore the notebooks:</strong>
        <p>Navigate to the <code>nlp/</code> or <code>computer_vision/</code> directories to run the Jupyter Notebooks and explore each classification approach.</p>
    </li>
</ol>
