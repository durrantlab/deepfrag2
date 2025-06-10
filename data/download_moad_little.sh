# This downloads a small subset of the MOAD dataset. Good for testing.

mkdir -p moad_little

# Note the link below is not guaranteed to be permanent.
curl -O https://durrantlab.pitt.edu/tmp/moad_little.zip

unzip moad_little.zip
rm moad_little.zip
