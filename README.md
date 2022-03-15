# banana-penguin
Repo for few-shot learning evaluation. 

Run the command to add to path :
```
 export PYTHONPATH=<path>/banana-penguin:$PYTHONPATH
```

Examples:
```
    python run_closest_summet.py --features-path \
        "['<path>/miniAS50backbone11.pt', '<path>/miniAS100backbone11.pt', '<path>/miniAS150backbone11.pt', '<path>/miniAS200backbone11.pt']" \
        --features-base-path '<path>/minifeaturesAS1.pt11' \
        --centroids-file '<path>/miniImagenetAS200noPrepLamda05.pickle' --lamda-mix 0 --n-runs 100000 --preprocessing 'ME';
```