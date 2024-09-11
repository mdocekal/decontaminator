# decontaminator
Tool for decontamination of training datasets

## Installation
Use the standard requirements.txt for installation of other packages:

    pip install -r requirements.txt

Also, as there are Cython extensions, you need to build them:
    
    python setup.py build_ext --inplace

or use:
    
    ./build_cython.sh

You can also install it as a package:

    pip install .

## Decontamination of test datasets

### 1. Extract ngrams from test datasets

At first, we need to gather all ngrams in our test sets and obtain in which samples they are present. We can do it by running the following command:

    ./run.py run.py make_ngram_map hynky/klokan-qa klokan_qa_map.json 13 --dataset_config "balanced" --format_str "{{question}}" --split "test" --allow_shorter --hf_cache "hf_cache"

This command will create a file `klokan_qa_map.json` with 13grams and their samples from HuggingFace dataset `hynky/klokan-qa`. It will work with the `balanced` configuration of the dataset and will use the `question` field for the ngram extraction. The `--split` selects test split. The `--allow_shorter` flag allows extraction of shorter ngrams when the string is shorter than the ngram size.  Finally, the `--hf_cache` specifies the directory for the Hugging Face cache.

See help for more information about arguments

Run this command for all datasets you want to decontaminate.

### 2. Merge ngram maps

In case you have multiple ngram maps, you can merge them by running the following command:

    ./run.py merge_map *.json --output merged_ngrams.json

Warning: This will change the original sample indices to the global indices in order to have unique indices for all samples.

### 3. Search contaminated
In this step we will use training dataset to identify contaminated test samples and ngrams. We will use the following command:

    ./run.py search_contaminated train.jsonl klokan_qa_map.json contaminated_indices.json contaminated_ngrams.json --field "text" --ignore_above 10 --workers -1

This command will search for contaminated test samples and ngrams in `klokan_qa_map.json` by given `train.jsonl`. The results will be saved in `contaminated_indices.json` and `contaminated_ngrams.json`. The `--field` specifies the field in the train dataset that will be used for the search. The `--ignore_above` allows to ignore frequent ngrams (frequent in given map, thus in test sets). The `--workers` specifies the number of parallel workers used for the search.

#### Back conversion of indices
In case you are working with merged ngram maps, you might need to convert indices back to their original form. You can do it by running the following command:

    ./run.py indices_2_dataset_indices contaminated_indices.json merged_ngrams.json contaminated_indices_per_dataset.json

#### Per dataset sort of ngrams
Another useful tool, when working with multiple datasets, is to sort contaminated ngrams into their original datasets and see
which ngrams and samples are contaminated in each dataset. You can do it by running the following command:

    ./run.py contaminated_ngrams_per_dataset contaminated_ngrams.json merged_ngrams.json contaminated_ngrams_per_dataset.json

It will create a map like this:

```json
{
    "dataset b": {
        "ngram": [original sample indices]
    }, 
    "dataset A": {
        "ngram": [original sample indices]
    }
}
```
