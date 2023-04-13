# If the environmental variable MOAD_DIR is not defined, source env.sh

if [ -z "$MOAD_DIR" ]; then
    echo "MOAD_DIR is not defined. Sourcing env.sh to define..."

    echo "Loading modules and environment. Also, define PYTHON_EXEC, EVERY_CSV_BSNM (usually 'every.csv')"
    echo "and MOAD_DIR variables. You must put all this in the env.sh file (not"
    echo "synced to git)"

    . env.sh
fi

# Get python script path
MAIN_DF2_PY=`realpath $(ls ../MainDF2.py)`
