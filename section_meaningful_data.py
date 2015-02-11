import csv

long_filename = 'meaningful_features.csv'

short_filename_1 = 'meaningful_features_100000.csv'
short_filename_2 = 'meaningful_features_200000.csv'
short_filename_3 = 'meaningful_features_300000.csv'
short_filename_4 = 'meaningful_features_400000.csv'
short_filename_5 = 'meaningful_features_500000.csv'

input_file = open(long_filename,'r')
output_file_1 = open(short_filename_1,'w')
output_file_2 = open(short_filename_2,'w')
output_file_3 = open(short_filename_3,'w')
output_file_4 = open(short_filename_4,'w')
output_file_5 = open(short_filename_5,'w')

input_csv = csv.reader(input_file, delimiter=',', quotechar='"')
output_csv_1 = csv.writer(output_file_1, delimiter=',', quotechar='"')
output_csv_2 = csv.writer(output_file_2, delimiter=',', quotechar='"')
output_csv_3 = csv.writer(output_file_3, delimiter=',', quotechar='"')
output_csv_4 = csv.writer(output_file_4, delimiter=',', quotechar='"')
output_csv_5 = csv.writer(output_file_5, delimiter=',', quotechar='"')

count = 1 
for row in input_csv:
	if count > 500000:
		break
	elif count <= 100000:
		output_csv_1.writerow(row)
	elif count <= 200000:
		output_csv_2.writerow(row)
	elif count <= 300000:
		output_csv_3.writerow(row)
	elif count <= 400000:
		output_csv_4.writerow(row)
	elif count <= 500000:
		output_csv_5.writerow(row)
	count += 1

input_file.close
output_file_1.close
output_file_2.close
output_file_3.close
output_file_4.close
output_file_5.close


