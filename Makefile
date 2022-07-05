PARCIALITO=1 2 3

all: $(PARCIALITO)

$(PARCIALITO):
	$(eval DIR = Parcialito$@)
	jupyter nbconvert --to pdf --execute $(DIR)/parcialito-$@.ipynb
	pdfunite $(DIR)/enunciado.pdf $(DIR)/parcialito-$@.pdf "$(DIR)/P$@ - 100029.pdf"

