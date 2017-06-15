

# matplotlib error

The tests might not work with OSX because of some funky matploblib behavior. When the testing script tries to import matplotlib you might get an error: " RuntimeError: Python is not installed as a framework...See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information". See this [stackoverflow post](https://stackoverflow.com/questions/29433824/unable-to-import-matplotlib-pyplot-as-plt-in-virtualenv) .

The solution is to make a file called **matplotlibrc** in the directory **~/.matplotlib** then add the line **backend: TkAgg**. In particular:

`cd User/YOURNAME/.matplotlib`

`vi matplotlibrc`

then add the line: **backend: TkAgg**
