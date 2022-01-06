export CUR_APP=`cat /working/cur_app_name.txt`
# echo "cd /mnt/apps/${CUR_APP}/"
cd /mnt/apps/${CUR_APP}/
./run.sh

# Run bash at end so user can inspect.
# /usr/bin/bash