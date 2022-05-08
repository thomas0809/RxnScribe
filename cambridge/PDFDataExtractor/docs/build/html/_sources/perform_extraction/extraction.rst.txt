Default extraction
==================
PDFDataExtractor automatically choose the template to use to perform the extraction.

Text Mode
=========
PDFDataExtractor outputs pure text by default, following below to use this feature: 


	.. code-block:: python

		# Import PDFDataExtractor
		from pdfdataextractor import Reader

		# Spefify the path to the PDF file
		path = r'the path to the PDF file'
		
		# Create an instance
		file = Reader()
		
		# Read the file
		pdf = file.read_file(path)
		
		# Get pure text
		pdf.plaintext()

Semantic Mode
=============
PDFDataExtractor can also output results as semantic information, following below to use this feature: 


	.. code-block:: python

		# Import PDFDataExtractor
		from pdfdataextractor import Reader

		# Spefify the path to the PDF file
		path = r'the path to the PDF file'

		# Create an instance
		file = Reader()

		# Read the file
		pdf = file.read_file(path)

		# Get Caption
		pdf.caption()

		# Get Keywords
		pdf.keywords()
		
		# Get Title
		pdf.title()

		# Get DOI
		pdf.doi()

		# Get Abstract
		pdf.abstract()

		# Get Journal
		pdf.journal()

		# Get Journal Name
		pdf.journal('name')

		# Get Journal Year
		pdf.journal('year')

		# Get Journal Volume
		pdf.journal('volume')

		# Get Journal Page
		pdf.journal('page')

		# Get Section titles and corresponding text
		pdf.section()

		# Get References
		pdf.reference()


Image Mode (Temporarily unavailable)
====================================
PDFDataExtractor can also export images in the PDF, following below to use this feature: 

	.. code-block:: python

		# Import PDFDataExtractor
		from pdfdataextractor import Reader

		# Spefify the path to the PDF file
		path = r'the path to the PDF file'

		# Create an instance
		file = Reader()

		# Read the file
		pdf = file.read_file(path)
		
		# To access a specific image
		pdf.iamge()[0]

Known Issues
============

**In ACS**

* In ACS, a few journals have two section title styles existing at the same time, namely: numbered one and ■ one. This could confuse the title filtration function because two styles have largely different font sizes. But this won’t affect reference extraction
* Reference extracted might not be in order
* Parts of extracted reference could be missing
		
**In Elesvier**

* Potentially weak journal extraction leads to missing journal information
* Unnumbered references can be messy
		
**In RSC**

* Title can be missing
* Journal year, volume and page numbers can be missing in certain articles
* Some section titles can be missed but reference section remains solid
		
**In Advanced Family**

* Reference entries can be mixed
* Keywords can be found inside reference entries, roughly 1 in 20
* Some authors place their bio at the very end, such words are not excluded from reference at the moment
		
**In CAEJ**

* Keywords can be incomplete
* In AngewandteKeywords might not be in order
