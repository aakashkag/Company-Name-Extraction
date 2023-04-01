import spacy
def test_saved_model(output_dir):
	TESTING_DATA= [
		'© COPYRIGHT 2019 AFR FURNITURE RENTAL',
		'© 2019 All Rights Reserved by Abundant Life Worship Center of Whippan',
		'Copyright ©Voyagers Travel',
		'ABVNBYOND Inc © 2013 | All Rights Reserved',
                'Munich Re supplies insurance to other companies',
                'Mind the Math LLC suplies software to Mind the Grow LLC for various purposes.',
	]
	# test the saved model
	print("Loading from", output_dir)
	nlp2 = spacy.load(output_dir)
	for text in TESTING_DATA:
		doc = nlp2(text)
		print(f'\n Input Text:{text} \n Extracted Entities', [(ent.text, ent.label_) for ent in doc.ents])
		

test_saved_model('./Model')
