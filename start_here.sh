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
    year=2022
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
    $run_command python top2vec_model.py -c -t --year 2022
    $run_command python top2vec_model.py -c -t --search ${searches[*]}
elif [ -n "$top2vec" ]; then
    $run_command python top2vec_model.py --year 2022
    $run_command python top2vec_model.py --search ${searches[*]}
fi

if [ -n "$doc2map" ]; then
    $run_command python doc2map.py --year 2022
    # $run_command python doc2map.py
fi
