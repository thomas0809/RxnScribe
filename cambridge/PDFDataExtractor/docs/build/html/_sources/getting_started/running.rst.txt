Run PDFDataExtractor
====================
You can start to extract information from PDF files once the tool is correctly installed.

Import necessary modules
------------------------

	.. code-block:: python
	
		from pdfdataextractor import Reader

Pass a PDF file
---------------

	.. code-block:: python

		# Spefify the path to the PDF file 
		path = r'the path to the PDF file'

		# Create an instance
		file = Reader()

		# Read the file
		pdf = file.read_file(path)
		
		# Test if pdf is returned successfully
		pdf.test()

