grep -A 5 "avgOfCheckpoints" test_results-1.json  | grep cosine | awk '{print $2}' | sed "s/,//g" | awk '{t=t+$1; n=n+1; print(t/n)}' | tail -n 1 | tee average_cosine_sim_to_closest_match.txt
