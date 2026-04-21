# GAI4 Course

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/MustafaArikan/GAI4_course.git
cd GAI4_course
```

### 2. Install dependencies

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the environment for the unit you want to work on:

```bash
cd Unit_0
uv sync
```

## Datasets

- **UC Merced Land Use** — 21-class aerial/satellite imagery. Yang & Newsam, *ACM SIGSPATIAL GIS*, 2010. [blanchon/UC_Merced on HuggingFace](https://huggingface.co/datasets/blanchon/UC_Merced)
- **Fruits-360** — 360+ fruit/vegetable classes, clean backgrounds. Mureşan & Oltean, *Acta Univ. Sapientiae, Informatica*, 2018. [fruits-360 on Kaggle](https://www.kaggle.com/datasets/moltean/fruits)
- **FIDS30** — 30 fruit classes, in-the-wild images. Škrjanec, *Univ. of Ljubljana*, 2013. [vicos.si/resources/fids30](https://www.vicos.si/resources/fids30/)
- **MNIST** — 10-class handwritten digits. LeCun, Cortes & Burges, 1998. [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/)
- **Fashion-MNIST** — 10-class clothing images, drop-in MNIST replacement. Xiao, Rasul & Vollgraf, *arXiv:1708.07747*, 2017. [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
