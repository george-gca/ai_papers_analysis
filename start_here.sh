#!/bin/bash
# Enable poetry if not running inside docker and poetry is installed
if [[ $HOSTNAME != "docker-"* ]] && (hash poetry 2>/dev/null); then
    run_command="poetry run"
fi

# cluster_conferences=1
# find_words_usage_over_conf=1
# n_clusters=100
# train_top2vec=1
# top2vec=1
doc2map=1
year=2022

searches=(
      "bert"
      "bert_based"
      "capsule"
      "catastrophic_forgetting"
      "continual_learning"
      "explainability"
      "explanatory_interactive_learning"
      "incremental_learning"
      "interactive_learning"
      "interpretability"
      "large_scale pre_training"
      "model_editing"
      "multimodal_dataset"
      "multimodal_feature"
      "multimodal pre_training"
      "new_dataset"
      "new_multimodal_dataset"
      "pre_training"
      "rationale"
      "representation_learning"
      "scene_graph"
      "survey"
      "transformer"
      "visual_question_answering"
      "visual_question_answering new_dataset"
      "visual_reasoning"
      "vqa"
    )


conferences_years=(
    "aaai 2017"
    "aaai 2018"
    "aaai 2019"
    "aaai 2020"
    "aaai 2021"
    "aaai 2022"
    "acl 2017"
    "acl 2018"
    "acl 2019"
    "acl 2020"
    "acl 2021"
    "acl 2022"
    "coling 2018"
    "coling 2020"
    "coling 2022"
    "cvpr 2017"
    "cvpr 2018"
    "cvpr 2019"
    "cvpr 2020"
    "cvpr 2021"
    "cvpr 2022"
    "eacl 2017"
    "eacl 2021"
    "eccv 2018"
    "eccv 2020"
    "eccv 2022"
    "emnlp 2017"
    "emnlp 2018"
    "emnlp 2019"
    "emnlp 2020"
    "emnlp 2021"
    "findings 2020"
    "findings 2021"
    "findings 2022"
    "iccv 2017"
    "iccv 2019"
    "iccv 2021"
    "iclr 2018"
    "iclr 2019"
    "iclr 2020"
    "iclr 2021"
    "iclr 2022"
    "icml 2017"
    "icml 2018"
    "icml 2019"
    "icml 2020"
    "icml 2021"
    "icml 2022"
    "ijcai 2017"
    "ijcai 2018"
    "ijcai 2019"
    "ijcai 2020"
    "ijcai 2021"
    "ijcai 2022"
    "ijcnlp 2017"
    "ijcnlp 2019"
    "ijcnlp 2021"
    "kdd 2017"
    "kdd 2018"
    "kdd 2020"
    "kdd 2021"
    "naacl 2018"
    "naacl 2019"
    "naacl 2021"
    "naacl 2022"
    "neurips 2017"
    "neurips 2018"
    "neurips 2019"
    "neurips 2020"
    "neurips 2021"
    "neurips 2022"
    "neurips_workshop 2019"
    "neurips_workshop 2020"
    "neurips_workshop 2021"
    "neurips_workshop 2022"
    "sigchi 2018"
    "sigchi 2019"
    "sigchi 2020"
    "sigchi 2021"
    "sigchi 2022"
    "sigdial 2017"
    "sigdial 2018"
    "sigdial 2019"
    "sigdial 2020"
    "sigdial 2021"
    "tacl 2017"
    "tacl 2018"
    "tacl 2019"
    "tacl 2020"
    "tacl 2021"
    "tacl 2022"
    "wacv 2020"
    "wacv 2021"
    "wacv 2022"
)


# https://github.com/awslabs/autogluon/issues/1020#issuecomment-926089808
export OPENBLAS_NUM_THREADS=15
export GOTO_NUM_THREADS=15
export OMP_NUM_THREADS=15

if [ -n "$cluster_conferences" ]; then
    echo -e "\nClustering conferences' words"
    $run_command python cluster_conference_words.py --clusters $n_clusters --word_dim 3 -l info

    echo -e "\nClustering conferences' papers"
    $run_command python cluster_conference_papers.py --clusters $n_clusters --paper_dim 3 -l info

    echo -e "\nClustering search results"
    for search in "${searches[@]}"; do
    	$run_command python cluster_filtered_papers.py "$search" -l info --name "$search" --clusters 10 -p 3 --suffix _50000w_150_clusters_pwc -y $year
    	$run_command python cluster_filtered_papers.py "$search" -l info --name "$search" --clusters 30 -p 3 --suffix _50000w_150_clusters_pwc
    done
fi

if [ -n "$find_words_usage_over_conf" ]; then
    conferences=(
        "aaai"
        "acl"
        "coling"
        "cvpr"
        "eccv"
        "emnlp"
        "iccv"
        "iclr"
        "icml"
        "ijcai"
        "ijcnlp"
        "kdd"
        "naacl"
        "neurips"
        "sigchi"
        "sigdial"
        "tacl"
        "wacv"
    )
    for conference in "${conferences[@]}"; do
        $run_command python find_words_usage.py --suffix _50000w_150_clusters_pwc -c $conference
    done
fi

if [ -n "$train_top2vec" ]; then
    $run_command python top2vec_model.py -c -t --year $year
    $run_command python top2vec_model.py -c -t --search ${searches[*]}
elif [ -n "$top2vec" ]; then
    $run_command python top2vec_model.py --year $year
    $run_command python top2vec_model.py --search ${searches[*]}
fi

if [ -n "$doc2map" ]; then
    for conference in "${conferences_years[@]}"; do
        conf_year=($conference)
        if [[ "${conf_year[1]}" == "$year" ]]; then
            $run_command python doc2map.py --year $year --conference ${conf_year[0]}
        fi
    done

    $run_command python doc2map.py --year $year
fi
