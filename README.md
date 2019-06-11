

Analyzing the Perceived Severity of Cybersecurity Threats Reported on Social Media
====================

This repository contains the code and resources from the following paper:

Analyzing the Perceived Severity of Cybersecurity Threats Reported on Social Media

Shi Zong, Alan Ritter, Graham Mueller, Evan Wright

https://www.aclweb.org/anthology/papers/N/N19/N19-1140/

### Demo

See demo at:

http://kb1.cse.ohio-state.edu:8123/events/threat

### Python version

python3.6

### Dataset preparation

#### Get tweets

We provide a file tweet_anno_id.tsv under dataset containing tweets ids we have annotated. This file contains two columns: tweets ids and timestamp.

Actual tweet contents can be acquired by using Semeval Twitter data download script (https://github.com/aritter/twitter_download). Please follow instructions there to get tweets via Twitter API.

Once you get all tweets downloaded, run prepare_dataset.py under dataset folder to recover the dataset we are using.

```
 python prepare_dataset_for_tagging.py 
```

#### Tweets tokenization

We use Twitter NLP (https://github.com/aritter/twitter_nlp) for tokenization.
 
We suggest using tagging tool in following way, which reads in json line format files and directly appends  'tags' field into the original file.

```
cat ./dataset.json | python python/ner/extractEntities2_json.py > dataset_tagged.json
```

#### Dataset preparation

There are some final steps to get our annotated dataset. Specifically, we need to replace entities extracted by tagging tool with a special token <TARGET>.

```
python dataset_processing.py PATH_TO_TAGGED_FILE
```

#### Note

(1) We notice some tweets are marked as "Not Available" when downloading through twitter API. We can not directly release our dataset with respect to Twitter's privacy policy.

(2) For annotated tweets, we have specified the entity along with location we want to replace. For your own data, you could use getEntitySegClass() and replaceEntityTarget() in utils.tagging_process.py.

(3) We recommend replacing all digits with 0.

### Feed your own data

#### Train model

We have provided our pre-trained model under trained_model directory. For threat existence classifier, we use 4,000 annotated tweets. For threat severity classifier, we use 1,200 annotated tweets.

#### Input data format
Input data should be in .json format. We provide a sample input file sample_input.json for your reference. The classifier looks for 'text_TARGET' field.

You could use writeJSONFile() in utils.io py for generating files.

#### Calculate scores

(1) for threat existence classifier

```
 python main.py existence PATH_TO_YOUR_DATA.json
```

(2) for threat severity classifier

```
 python main.py severity PATH_TO_YOUR_DATA.json
```

 Prediction output file is stored under the same directory of your input data, with file name PATH_TO_YOUR_DATA_with_score.json. Prediction scores are in 'existence_prob' (or 'severity_prob') field.

### Train model

If you want to train your own model, first change save_model to True and then

(1) for threat existence classifier

```
 python train_model_lr.py existence PATH_TO_TRAINING_DATA.json
```

(2) for threat severity classifier

```
 python train_model_lr.py severity PATH_TO_TRAINING_DATA.json
```

You could change n-gram window size by using -w flag. Please make sure you are using the same n-gram window size for both training and evaluation.

### Reference

@inproceedings{zong-etal-2019-analyzing,
    title = "Analyzing the Perceived Severity of Cybersecurity Threats Reported on Social Media",
    author = "Zong, Shi  and
      Ritter, Alan  and
      Mueller, Graham  and
      Wright, Evan",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1140",
    pages = "1380--1390",
    abstract = "Breaking cybersecurity events are shared across a range of websites, including security blogs (FireEye, Kaspersky, etc.), in addition to social media platforms such as Facebook and Twitter. In this paper, we investigate methods to analyze the severity of cybersecurity threats based on the language that is used to describe them online. A corpus of 6,000 tweets describing software vulnerabilities is annotated with authors{'} opinions toward their severity. We show that our corpus supports the development of automatic classifiers with high precision for this task. Furthermore, we demonstrate the value of analyzing users{'} opinions about the severity of threats reported online as an early indicator of important software vulnerabilities. We present a simple, yet effective method for linking software vulnerabilities reported in tweets to Common Vulnerabilities and Exposures (CVEs) in the National Vulnerability Database (NVD). Using our predicted severity scores, we show that it is possible to achieve a Precision@50 of 0.86 when forecasting high severity vulnerabilities, significantly outperforming a baseline that is based on tweet volume. Finally we show how reports of severe vulnerabilities online are predictive of real-world exploits.",
}
