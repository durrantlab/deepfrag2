mkdir -p moad
cd moad
curl -O http://bindingmoad.org/files/biou/every_part_a.zip
curl -O http://bindingmoad.org/files/biou/every_part_b.zip
curl -O http://bindingmoad.org/files/csv/every.csv

unzip every_part_a.zip
unzip every_part_b.zip

rm every_part_?.zip


