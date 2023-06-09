# Complete Data Science BootCamp
<p align="center">
   <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/Data%20Science.png" alt="Data Science" width="1000" height="500">
</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#numpy">Numpy</a>
    </li>
    <li><a href="#pandas">Pandas</a></li>
    <li><a href="#data_visualization">Data Visualization</a></li>
    <li><a href="#data_preprocessing">Data Preprocessing</a></li>
    <li><a href="#Machine Learning">Machine Learning</a></li>
    <li><a href="#Cloud Computing for Machine Learning">Cloud Computing for Machine Learning</a></li>
    <li><a href="#Deep Learning">Deep Learning</a></li>
    <li><a href="#Projects">Projects</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li> 
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## 📋 About The Project

<p align="justify"> The Complete Data Science Bootcamp project is an intensive training program designed to equip participants with skills and knowledge in the field of data science. In this project, participants will learn basic data science concepts, including data processing, statistical analysis, and data modeling. In addition, participants will also minimize the use of the latest technologies and tools in data science, such as Python, SQL, machine learning, and data visualization. The Complete Data Science Bootcamp project aims to prepare participants to become professional data scientists who are reliable and ready to compete in an increasingly competitive job market. </p>

<!-- TECHNOLOGIES -->
## 🛠 Technologies

The following technologies were used in this project:

* Python
* NumPy
* Pandas
* Matplotlib
* Jupyter

<!-- Numpy -->
## 🐍 Numpy

<p align="justify"> NumPy (Numerical Python) is a Python library for scientific computing that provides a powerful array processing capability. NumPy provides an efficient implementation of multi-dimensional arrays, which are the main data structure for numerical computing in Python. using NumPy enables users to process numeric data efficiently and effectively, as well as accelerate the development of code and applications that require processing of numeric data. </p>

1. Numpy Arrays

   * Zero D Arrays
   
     <p align="justify"> Zero-dimensional arrays, also known as scalar arrays, are arrays with no axes or dimensions. They are used to represent a single scalar value, such as a constant or a single data point. </p>
     
     ```sh
     import numpy as np
     a = np.array(42)
     print(a) #42
     b = np.array([1,2,3,4])
     print(b.ndim) # 1 --> ndim is used to count the number of arrays
     ```
   * 2D Arrays
   
     <p align="justify"> 2D arrays, also known as matrices, are arrays with two axes or dimensions. They are used to represent tabular data or a matrix of values, such as an image with height and width. In NumPy, 2D arrays can be created using the np.array() function and passing a list of lists as the input.</p>
     
     ```sh
     import numpy as np
     a = np.array([[1,2,3],[4,5,6]])
     print(a) #[[1,2,3],
              #[4,5,6]]
     print(a.ndim) #2 --> ndim is used to count the number of arrays
     ```
   * 3D Arrays
   
     <p align="justify"> Three-dimensional (3D) arrays, also known as a cube of data, are arrays with three axes or dimensions. They are used to represent data that can be thought of as a cube or a rectangular prism, such as RGB images or volume data. Each element in a 3D array is accessed using three indices, corresponding to the three axes of the cube. </p>
     
     ```sh
     import numpy as np
     a = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
     print(a) #[[[1,2,3],
                #[4,5,6]],
                #[[1,2,3],
                #[4,5,6]]]
     print(a.ndim) #3 --> ndim is used to count the number of arrays
     ```  
2. Arrays Shape and Reshape
   
   * Shape
     <p align="justify"> "Shape" in NumPy refers to the size or dimensions of an array. For one-dimensional arrays, the shape only contains a single integer representing the length of the array. For multi-dimensional arrays, the shape contains a tuple of integers representing the size of the array in each dimension. For example, a shape of (2, 3) would indicate a two-dimensional array with 2 elements in the first dimension and 3 elements in the second dimension. </p>
      
      ```sh
       import numpy as np
       a = np.array([1,2,3,4,5])
       print(a.shape) #(5,)
       b = np.array([1,2,3,4,5],[[6,7,8,9,10]])
       print(b.shape) #(2,5) containing the elements 1 through 10 arranged in 2 rows and 5 columns
       ```
    * Reshape
    
      <p align="justify"> Reshape is a method in NumPy for changing the dimensions or shape of an array without changing its data. By using the reshape method, we can convert a one-dimensional array to a multi-dimensional array, or reshape a multi-dimensional array to a different shape with the same number of elements. For example, we can reshape a one-dimensional array with 12 elements to a two-dimensional array with a size of 3x4, or a two-dimensional array with a size of 2x6 to a three-dimensional array with a size of 3x2x2.</p>
      
      ```sh
      import numpy as np
      a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
      newarray = a.reshape(3,4)
      print(newarray) #[[1,2,3,4],
                      #[5,6,7,8],
                      #[9,10,11,12] #containing the elements 1 through 12 arranged in 3 rows and 4 columns
      
      newarray1 = a.reshape9(2,3,2)
      print(newarray1) #[[[1,2],
                         #[3,4],
                         #[5,6]],
                        
                         #[[7,8],
                         #[9,10],
                         #[11,12]]]
      #So the array would have 2 blocks, each with 3 rows and 2 columns. The elements in the first block 
      #would be 1, 2, 3, 4, 5, and 6, while the elements in the second block would be 7, 8, 9, 10, 11, and 12.
      
      b = np.array([[1,2,3],[4,5,6]])
      c = a.reshape(-1)
      prinr(c) #([1,2,3,4,5,6]) #reshape(-1) is used to reshape an array into one dimension
      ```
3. Arrays Indexing

    <p align="justify"> Array indexing is a way to access specific elements or a subset of elements from a NumPy array. In NumPy, indexing starts at 0, which means that the first element in an array has an index of 0, the second element has an index of 1, and so on. </p>
    
   ```sh
   import numpy as np
   a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
   print(a[5]) #[6] five element in an array has an index of 6
      
   b = np.array([[1,2,3],[4,5,6]])
   print(b[0,2]) #3 Index 0 in column 1 is equal to 1, so column 1 index 2 is equal to 3. 
      
   c = np.array([[1,2,3],[4,5,6]])
   print(c[1,0]) #4 Index 1 in column 1 is equal to 2, so column 2 index 0 is equal to 4.
   ```
5. Array Iterating

     <p align="justify">Array iteration is the process of finding each element of an array in sequence to perform a specific operation. In NumPy, we can perform array iteration using a for loop. Each element of the array can be accessed using array indexing. For example, for a two-dimensional array, we can use two loops to access each element. Then, we can perform any operation on each element as needed.</p>
     
    ```sh
    import numpy as np
    a = np.array([1,2,3,4,5])
    for i in a:
    print(i) #1
             #2
             #3
             #4
             #5
    b = np.array([[1,2,3],[4,5,6]])
    for i in b:
    print(i) 
             #[1,2,3],
             #[4,5,6]
    for i in b:
        for j in i:
    print(j) #1
             #2
             #3
             #4
             #5
             #6
    ```
6. Silicing

     <p align="justify"> Slicing arrays is a technique for taking a portion or subset of a larger array by selecting only certain elements within that array. In NumPy, slicing is done using square brackets [ ] and the range of array indices that we want to take. For example, to take a portion of an array from index 3 up to index 6, we can use the syntax [3:7]. Additionally, we can also take elements with a certain step size using the syntax [start: end: step]. For instance, to take every second element from an array, we can use the syntax [::2].</p>
    
    ```sh
    import numpy as np
    a = np.array([1,2,3,4,5])
    print(a[1:3]) #2,3,4
    print(a[2:])  #3,4,5
    print(a[:3])  #2,3,4
    print(a[1:4:2]) #2,4 Index 1 = 2 and index 4 = 5, so the selected indexes are from index 1 to 4 with a step of 2.
       
    b = np.array([[1,2,3,4],[5,6,7,8]])
    print(b[1,1:3]) #6,7,8 Column with index 1 is [5, 6, 7, 8], so the elements with index 1 to index 3 are 6, 7, and 8.
    print(b[0,1:3]) #2,3,4 Column with index 0 is [1, 2, 3, 4], so the elements with index 1 to index 3 are 2, 3, and 4.
    ```
    
7. Array Searching and Shorrting

    <p align="justify"> Searching and sorting arrays refer to finding specific values or ordering the elements of an array in a specific manner. In NumPy, we can use various methods to search for elements in an array, such as np.where() and np.searchsorted(). We can also sort the elements of an array in ascending or descending order using np.sort() and np.argsort(). Additionally, we can sort the elements of an array based on a specific criterion using np.lexsort() and np.partition(). </p>
   
   ```sh  
   import numpy as np
   a = np.array([1,2,3,4,5,1,4,5,1])
   x = np.where(a == 1)
   print(x) # (array([0,5,8],dtype = int64),)
   
   y = np.where(a % 2 == 0)
   print(y) # (array([1,3,6],dtype = int64),)
       
   z = np.short(a)
   print(z) # (array([1,1,1,2,3,4,4,5,5])
   ```
<!-- Numpy -->
## 🧰 Pandas

<p align="justify"> Pandas is one of the Python libraries used for data analysis and data manipulation. This library provides flexible and efficient data structures for working with structured data such as SQL tables or Excel spreadsheets. Pandas allows users to perform many tasks, such as reading and writing data in various formats, such as CSV, Excel, SQL, and others. Additionally, pandas enables users to transform data into forms suitable for analysis, such as filtering, grouping, and aggregating data. This library also has the ability to perform attractive data visualization. 
Pandas is highly popular among data scientists, computer scientists, and engineers because it enables them to quickly and efficiently retrieve, manipulate, and analyze data. Additionally, pandas has excellent documentation and an active community, making it easy to find help and solutions when encountering problems or difficulties in using this library.</p>

1. Series in Pandas
   
   <p align="justify"> Pandas Series is a one-dimensional data structure similar to an array but with user-defined index capabilities. Pandas Series can contain numeric data, string data, Python objects, and other data. A set of Pandas Series can also form a Pandas DataFrame. Pandas Series can be created from a Python list, NumPy array, or dictionary. In addition, we can perform various operations on Pandas Series, such as filtering, grouping, aggregation, and data visualization.
   Pandas Series are very useful in data analysis and make data processing easier for various types of problems such as machine learning, data science, and business data analysis. Due to its index capabilities and comprehensive functionality, Pandas Series has become one of the most popular data structures in the Pandas library.</p>
   
   ```sh
   import pandas as pd
   a =[10,20,30,40,50]
   b = pd.Series(a)
   print(b) #0  10
            #1  20
            #2  30
            #3  40
            #4  50 dtype = int64
   c = {"one" :10,"second":100,"third":1000}
   d = pd.Series(c)
   print(d)#one      10
           #second  100
           #third  1000
    ```
2. Pandas DataFrame

   <p align="justify"> Pandas DataFrame is a two-dimensional data structure that is similar to SQL tables or Excel spreadsheets. Pandas DataFrame consists of rows and columns, with each column being able to have a different data type, such as numeric, string, or other Python objects. Each column in the DataFrame can be named, and the DataFrame can have user-defined indices.</p>
   
   ```sh
   import pandas as pd
   simple_data = {
          "Weekdays": ["Sunday","Monday","Tuesday"],
          "Scoree": ["80","90","100"]
          }
   d = pd.DataFrame(simpel_data)
   print(d)
   ```
    |       | Weekdays | Scoree   |
    |-------|----------|----------|
    |   0   |  Sunday  |    80    |
    |   1   |  Monday  |    90    |
    |   2   |  Tuesday |   100    |
    
3. Read CSV

   <p align="justify"> Read CSV is a function from the Pandas library in Python that is used to read files with CSV (Comma-Separated Values) format and convert them into a Pandas DataFrame. CSV file is a commonly used file format to store tabular data, such as data in an Excel spreadsheet. In a CSV file, each row represents a record, and each column contains data for different attributes in that record. The "read_csv" function allows us to read CSV files, load the data into a DataFrame, and perform various data manipulations on the file.</p>
   
   ```sh
      import pandas as pd
      df = pd.read_csv('airport.csv')
      print(df)
   ```
   <p align="center">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/csv.png" alt="Description_image" >
   </p>

4. Analyzing DataFrame in Pandas

   <p align="justify"> Analyzing DataFrame in Pandas involves performing various operations on the DataFrame to gain insights into the data, identify patterns, and make data-driven decisions.</p>
   Some of the common operations that can be performed on a Pandas DataFrame for analysis are:
   
   * Data Exploration - exploring the data to get an idea of its structure, quality, and distribution using various Pandas functions like head(), tail(), info(), describe(), etc.
   * Data Cleaning - cleaning the data by handling missing or null values, removing duplicates, correcting data types, etc.
   * Data Transformation - transforming the data to create new variables, combine variables, apply functions, filter data, etc.
   * Data Aggregation - aggregating the data to create summary statistics, group data by categories, pivot tables, etc.
   * Data Visualization - visualizing the data using various charts, plots, and graphs to identify patterns and relationships in the data.

   By performing these operations, we can gain a better understanding of the data and use it to make data-driven decisions. Pandas provides a rich set of functions and tools to perform these operations and analyze data efficiently.
   
   | Function | Description | Image |
   | :---: | :---: | :---: |
   | head() | Display the first five rows of the DataFrame | <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/head.PNG.png" alt="Head" width="1200" height="300"> |
   | tail() | Display the last five rows of the DataFrame | <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/tail.PNG.png" width="1200" height="300"> |
   | info() | Display general information about the DataFrame, such as the number of rows and columns, data types, and non-null values | <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/info.PNG.png" width="1200" height="300"> |
   | describe() | Display statistical summary of the DataFrame, such as mean, median, minimum and maximum values, etc | <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/describe.PNG.png" width="1200" height="300"> |


<!-- Data Visualization -->
## 🖥️ Data Visualization
1. Introduction to Matplotlib
   
   <p align="justify"> Matplotlib is a Python library used for creating visualizations, such as graphs, charts, and plots. It provides a flexible and powerful platform for data visualization, and is widely used in data science, scientific research, and engineering fields.
   Matplotlib allows users to create a variety of visualization types, including line plots, scatter plots, bar plots, histograms, and more. It also provides customization options for controlling aspects such as color, fonts, and sizes.</p>
   
   ```sh
   pip install matplotlib
   ```
   ```sh
   import matplotlib.pyplot as plt
   import numpy as np
       
   x = np.array([0,6])
   y = np.array([0,80])
   plt.plot(x,y)
   plt.show()   
   ```
   <p align="left">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/matplotlib.PNG" alt="Description_image" width="400" height="200">
   </p>

2. Different Type of Plots in Matplotlib
   * Line Chart 
      
      ```sh
     plt.plot()  
     ```
     ```sh    
     import matplotlib.pyplot as plt
     import numpy as np
       
     points = np.array([4,7,9,2,4,7,8])
     plt.plot(points)
     plt.show()   
     ```
     <p align="left">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/points.PNG" alt="Description_image" width="400" height="200">
     </p>
   
   * Bar Chart
     
     ```sh
     plt.bar()
     ```
     ```sh
     import matplotlib.pyplot as plt
     import numpy as np
       
     x = np.array(["a","b","c","d"])
     y = np.array([40,65,70,89])
     plt.bar(x,y)
     plt.show()   
     ```
     <p align="left">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/bar.PNG" alt="Description_image" width="400" height="200">
     </p>
   * Scatter Plot
     
     ```sh
     plt.scatter()
     ```
     ```sh
     import matplotlib.pyplot as plt
     import numpy as np
       
     x = np.array([63,43,30,23])
     y = np.array([40,65,70,89])
     plt.scatter(x,y)
     plt.show()   
     ```
     <p align="left">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/scatter.PNG" alt="Description_image" width="400" height="200">
     </p>
   * Pie Chart
     
     ```sh
     plt.pie()
     ```
     ```sh
     import matplotlib.pyplot as plt
     import numpy as np
       
     x = np.array([63,43,30,23])
     plt.pie(x)
     plt.show()   
     ```
     <p align="left">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/pie.PNG" alt="Description_image" width="400" height="200">
     </p>
   * Histogram Chart
     
     ```sh
     plt.hist()
     ```
     ```sh
     import matplotlib.pyplot as plt
     import numpy as np
       
     x = np.random.normal(170,10,250)
     plt.hist(x)
     plt.show()   
     ```
     <p align="left">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/histogram.PNG" alt="Description_image" width="400" height="200">
     </p>
   
3. Seaborn

   <p align="justify"> Seaborn is a Python data visualization library based on Matplotlib that provides a high-level interface for creating informative and attractive statistical graphics. Seaborn makes it easy to create a wide range of statistical visualizations, including scatter plots, line plots, bar plots, histograms, heatmaps, and more. Seaborn has several advantages over Matplotlib, including more attractive default styles, simpler syntax for common plot types, and built-in functionality for working with complex datasets. Seaborn also includes advanced features like built-in support for visualizing distributions, regression models, and statistical relationships between variables. Overall, Seaborn is a powerful tool for data scientists and analysts who need to quickly create visualizations that help them understand and communicate complex data. </p>
   
   * Boxplot
     
     ```sh
     import seaborn as sns
     import matplotlib.pyplot as plt
     import pandas as pd
       
     df=sns.load_dataset("tips")
     print(df)
     df.boxplot (by = 'day',column = ['total_bill'],grid = False)
     plt.show()
     ```
     <p align="center">
        <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/dataset%20tips.PNG" alt="Gambar 1" > &nbsp; &nbsp; &nbsp;
        <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/Boxplot.PNG" alt="Gambar 2" >
     </p>
     
   * Distplot
     
     ```sh
     import seaborn as sns
     import matplotlib.pyplot as plt
     import pandas as pd
       
     titanic = sns.load_dataset("titanic")
     titanic.head()
     df.boxplot (by = 'day',column = ['total_bill'],grid = False)
     plt.show()
      ```
     <p align="center">
        <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/datasheet%20titanic.PNG" alt="Gambar 1"  > &nbsp; &nbsp; &nbsp;
        <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/displot.PNG" alt="Gambar 2" >
     </p>

   * regplot
     
     ```sh
     import seaborn as sns
     import matplotlib.pyplot as plt
     import pandas as pd
       
     data = sns.load_dataset("mpg")
     data.head()
     sns.regplot(x='mpg',y='acceleration',data = data)
     plt.show()
      ```
     <p align="center">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/dataset%20mpg.PNG" alt="Description_image">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/regplot.PNG" alt="Description_image">
     </p> 

<!-- Data Preprocessing -->
## 🔬 Data Preprocessing
1. Handling Missing Values

   <p align="justify"> Handling missing values refers to the process of identifying and dealing with the absence of data in a dataset. Missing values can occur for various reasons, such as data entry errors, equipment failure, or participant dropout. Handling missing values is an important step in data preprocessing because most machine learning algorithms cannot handle missing values. </p>
   exemple : 
   
   ```sh
   import pandas as pd
   df = pd.read_csv('Placement_Data_Full_Class.csv')
   df()
   ```
   <p align="center">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/Placement.png" alt="Mising Value">
   </p>
   
   <p align="justify"> There are missing values in the salary column of the analyzed data. This indicates that there are employees whose salary information has not been inputted or incomplete data. Before conducting further analysis, it is necessary to fill in the missing values with an appropriate method such as using the mean, median, or mode, to ensure accurate and reliable analysis results. </p>
   
   ```sh
   df.isnull().sum()
   ```
   <p align="justify"> df.isnull().sum() is a code in Python used to count the number of missing or null values in each column of a DataFrame df. isnull() is a method used to mark each value in the DataFrame as either True if it is null, or False if it is not null. Then, sum() is used to count the number of True values for each column. Therefore, df.isnull().sum() will return a Series with the column index of the DataFrame df, and the values as the count of missing values in each column. This is very useful in data exploration to ensure that there are no missing data in certain columns of the DataFrame. </p>
   
   <p align="center">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/value%20null.png" alt="Mising Value">
   </p>
   
   ```sh
   df['salary'] = df['salary'].fillna(df['salary'].mode()[0])
   ```
   <p align="justify"> This line of code is filling missing values in the 'salary' column of a pandas DataFrame with the mode (most frequent value) of that column. The fillna() method of a pandas DataFrame is used to replace missing (NaN) values with a specified value. In this case, the specified value is the mode of the 'salary' column, which is obtained by calling the mode() method on the 'salary' column. The [0] index is used to extract the first (and only) value from the mode result, since the mode method can return multiple modes if there are ties.</p>
   
   <p align="justify"> This code assumes that the 'salary' column has missing values and that replacing them with the mode is an appropriate strategy for handling missing data in that column. It's important to note that filling missing values with the mode can introduce bias in the data if the mode is not representative of the underlying distribution. It's always a good practice to carefully consider different strategies for handling missing data and their potential impact on downstream analysis.</p>
   
   ```sh
   df.isnull().sum()
   ```
   <p align="center">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/dataisnotnull.png" alt="Mising Value">
   </p>
   
   ```sh
   df.isnull().sum()
   ```
   <p align="center">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/no%20missing%20values.png" alt="Mising Value">
   </p> 
   
    <p align="justify"> All 'salary' values have been filled and there are no more null values. The 'salary' column has been updated to fill all previously missing values. Null values in the 'salary' column have been replaced with the mode, resulting in a complete dataset with no missing values for 'salary'. </p>
    
2. Feature Enconding
   
   <p align="justify"> Feature encoding is the process of transforming categorical or text data into a numerical representation that can be used for machine learning algorithms. Machine learning algorithms typically work with numerical data, so feature encoding is necessary when working with non-numerical data. There are different types of feature encoding techniques, including: </p>
   
   * One-hot encoding
     
     <p align="justify"> This technique converts categorical data into a binary vector where each category is represented by a binary feature (0 or 1). For example, if we have a categorical feature "color" with possible values "red", "green", and "blue", we can one-hot encode it into three binary features: "is_red", "is_green", and "is_blue". </p>
     
     ```sh
     import pandas as pd
     df = pd.read_csv('type.csv')
     df.head(10)
     ```
     <p align="center">
        <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/enconding1.png" alt="Mising Value">
     </p>  
     
     ```sh
     df['type'].value_counts()
     ``` 
     <p align="center">
        <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/enconding2.png" alt="Mising Value">
     </p>  
     
     ```sh
     pd.get_dummies(df,columns=['type'])
     ``` 
     <p align="center">
        <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/enconding3.png" alt="Mising Value">
     </p> 
     
    * Label encoding
      
      <p align="justify"> This technique assigns a numerical label to each category in a categorical feature. For example, if we have a categorical feature "color" with possible values "red", "green", and "blue", we can label encode it as 1, 2, and 3, respectively.
   
    * Binary encoding 
      
      <p align="justify"> This technique converts a categorical feature into binary digits. Each category is assigned a binary code, and the binary codes are concatenated to form the encoded feature. For example, if we have a categorical feature "color" with possible values "red", "green", and "blue", we can binary encode it as follows: "red" = 00, "green" = 01, "blue" = 10. </p>
    
    * Count encoding
      
      <p align="justify"> This technique replaces each category with the number of times it appears in the dataset. For example, if we have a categorical feature "color" with possible values "red", "green", "blue", and "red", count encoding will assign the values 2, 1, and 1 to "red", "green", and "blue", respectively. </p>
    
    * Target encoding
      
      <p align="justify">This technique replaces each category with the mean value of the target variable for that category. For example, if we have a categorical feature "color" and a target variable "price", target encoding will assign the mean price for each color to that color's category. These techniques can be applied to different types of categorical data and may have different performance characteristics depending on the dataset and machine learning algorithm used. </p>

3. Feature Scealing
   
   <p align="justify">Feature scaling is a process in machine learning for normalizing or standardizing numerical features in a dataset. This is done to avoid differences in scale between features that can affect the performance of machine learning algorithms. For example, if the 'age' feature has a scale of 0-100, while the 'income' feature has a scale of 0-100000, then the machine learning algorithm will pay more attention to the 'income' feature because its value is much larger than 'age'. Therefore, we need to normalize or standardize the scale of features so that the scale of features is balanced and does not produce unwanted errors when performing machine learning. Some techniques used in feature scaling include Min-Max scaling (normalization), Z-score or Standardization, Log transform, and others. </p>

<!-- Machine Learning -->
## 🤖 Machine Learning

   <p align="justify"> Machine learning is a branch of artificial intelligence that teaches machines to learn from data without being explicitly programmed. Machines learn from data to find patterns and make predictions or decisions. The process of machine learning involves data collection, data preparation, model selection, model training, and model evaluation. The data collected must be well prepared before being fed into the model. Selecting the right model is also crucial for accurate results. There are three types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning involves using a dataset that is labeled or has correct answers. The model learns from this data to make accurate predictions or decisions when given new data. Unsupervised learning, on the other hand, does not involve labeled or correct answers, but rather attempts to find patterns in the data. Reinforcement learning involves providing feedback to the model to improve its results over time. Machine learning is used in various applications, including natural language processing, face recognition, image classification, and much more. In recent years, advancements in technology and the availability of big data have accelerated the development of machine learning and opened up many new opportunities.</p>
</p>

   1. Supervised Machine Learning
       
      <p align="justify"> Supervised machine learning is a type of machine learning where the model learns from labeled data to make predictions or decisions. In supervised learning, the dataset used for training the model contains input features (also known as independent variables) and corresponding target labels (also known as dependent variables or outputs). The goal of supervised learning is to train a model that can accurately map input features to their corresponding target labels, allowing it to make predictions on new, unseen data.</p>
    
   2. Unsupervised Machine Learning
      
      <p align="justify"> Unsupervised machine learning is a type of machine learning that involves training models on datasets without any labeled or pre-defined output variables. The goal is to find patterns or structures in the data without any specific guidance or supervision. In unsupervised learning, the model is fed with a dataset and it is expected to identify the underlying structure or relationships within the data. This can be done through various techniques such as clustering, dimensionality reduction, and association rule mining. 
      Unsupervised learning has many applications such as anomaly detection, market segmentation, and image segmentation. It is particularly useful when there is no prior knowledge of the data and the goal is to explore the underlying structure of the data. Some popular algorithms used in unsupervised learning include k-means clustering, hierarchical clustering, principal component analysis (PCA), and association rule mining. </p>
  
   3. Machine Learning Life Cycle
  
      The Machine Learning (ML) life cycle is a series of steps that a data scientist or ML engineer follows to create a successful ML model. These steps can vary depending on the specific problem or project, but generally, they include the following:</p>
      * Problem Definition: 
        
        The first step is to define the problem and determine what you want to achieve through machine learning. This involves identifying the business problem, defining the scope, and setting clear goals.
      * Data Collection: 
        
        The next step is to collect relevant data from various sources. This may involve web scraping, accessing APIs, gathering data from databases or spreadsheets, or other methods.
      * Data Preparation: 
        
        After collecting the data, it needs to be cleaned, pre-processed, and prepared for modeling. This may involve handling missing or incorrect values, transforming data, and selecting relevant features.
      * Data Splitting: 
        
        The prepared data is then divided into training, validation, and testing sets. This helps evaluate the model's performance and prevent overfitting.
      * Model Selection: 
        
        Once the data is prepared and split, you can select an appropriate model for the problem. This may involve choosing from a variety of algorithms and selecting the most suitable one for the specific problem.
      * Model Training: 
        
        The selected model is then trained on the training dataset. This involves fitting the model to the data and optimizing its parameters to achieve the best performance.
      * Model Evaluation: 
        
        The model is then evaluated on the validation dataset to determine its accuracy and performance. This step helps to fine-tune the model and optimize its parameters further.
      * Model Testing: 
        
        After the model has been optimized, it is tested on the testing dataset to assess its performance on unseen data. This step helps to determine the model's generalization capability.
      * Deployment: 
        
        Once the model is trained and tested, it can be deployed to production. This involves integrating the model into a larger system, creating an API, and providing access to end-users.
      * Monitoring and Maintenance: 
        
        After deployment, the model needs to be monitored and maintained regularly. This involves checking for any issues or bugs and updating the model as needed to improve its performance.
        
        These steps are iterative and require a lot of experimentation and fine-tuning to achieve the best results. The ML life cycle is an ongoing process, and models may need to be updated and retrained as new data becomes available or as the problem changes over time.
  
   4. Train Test Split
      
      <p align="justify"> Train test split is a technique in machine learning for dividing a dataset into two parts, namely the training set and the testing set. The training set is used to train the model, while the testing set is used to evaluate how well the model can make predictions on new data that it has not seen before. The dataset is split randomly with a certain proportion, such as 70% for the training set and 30% for the testing set. The proportion may vary depending on the amount of data available and the complexity of the model used.
      The purpose of train test split is to avoid overfitting, a condition where the model is too closely fitted to the training data and cannot make good predictions on new data. By separating the dataset into two parts, we can check the model's performance on data that was not used for training. Train test split can be done using libraries or modules in programming languages like Python, such as scikit-learn. The module provides functions to split the dataset with a certain proportion randomly. After the dataset is divided, we can train the model on the training set and evaluate its performance on the testing set. </p>
 
   5. Regression Analaysis
      
      <p align="justify"> Regression analysis is a statistical method used to examine the relationship between a dependent variable and one or more independent variables. The goal is to find a mathematical model that can predict the value of the dependent variable based on the values of the independent variables. Regression analysis can be used for both linear and nonlinear relationships. In linear regression, the goal is to find a straight line that best fits the data. In nonlinear regression, the goal is to find a curve or equation that best fits the data. There are different types of regression analysis, such as simple linear regression, multiple linear regression, logistic regression, and polynomial regression. Simple linear regression involves only one independent variable, while multiple linear regression involves two or more independent variables. Logistic regression is used when the dependent variable is categorical, and polynomial regression is used when there is a nonlinear relationship between the variables. </p> 
   
   6. Liner Regression 
     
       <p align="justify"> Linear regression is a type of regression analysis used in machine learning to predict a continuous outcome variable (also called a dependent variable) based on one or more predictor variables (also called independent variables). It assumes a linear relationship between the predictor variables and the outcome variable, meaning that the change in the outcome variable is proportional to the change in the predictor variables. In linear regression, the goal is to find the line of best fit that represents the relationship between the predictor variables and the outcome variable. The line of best fit is determined by minimizing the sum of the squared differences between the actual values of the outcome variable and the predicted values. </p>
     
      ```sh
      x = data_set.iloc[:,:-1].values
      y = data_set.iloc[:,1].values
      ``` 
      
      <p align="justify"> Generally, this code is used to separate the dataset into two variables, namely independent variable (x) and dependent variable (y) for further processing. The independent variable is the input variable used to predict the dependent variable, which is the output variable to be predicted. In general, the code is used to separate a dataset into two variables, namely the independent variable (x) and the dependent variable (y) for further processing. The independent variable is the input variable used to predict the dependent variable, which is the output variable that needs to be predicted. The first line of code, x = data_set.iloc[:,:-1].values, slices the dataset using .iloc[:,:-1], which means selecting all rows and all columns except the last column, which is the dependent variable column. The 'values' function is used to convert the sliced result into a Numpy array, so the resulting variable x contains all rows and all columns except the last column. The second line of code, y = data_set.iloc[:,1].values, slices the dataset using .iloc[:,1], which means selecting all rows in the second column, which is the dependent variable column. The 'values' function is used to convert the sliced result into a Numpy array, so the resulting variable y contains all rows in the second column (dependent variable). In the context of Machine Learning, the variable x will be used as an input to the Machine Learning model, while the variable y will be used as the output to be predicted by the model. </p> 
      
      ```sh
      from sklearn.model_selection import train_test_split
      x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
      ``` 
      <p align="justify"> The code is using the train_test_split function from the sklearn.model_selection module to split the dataset into training and testing data. The train_test_split function takes in the x and y variables, as well as a test_size parameter that determines the proportion of the data that will be used for testing. In this case, the test_size is set to 0.2, which means 20% of the data will be used for testing. The function returns four variables: x_train, x_test, y_train, and y_test. These variables represent the training and testing datasets for the independent and dependent variables. x_train and y_train represent the training data, which is used to train the machine learning model. x_test and y_test represent the testing data, which is used to evaluate the performance of the model on data that it has not seen before. The code assigns these variables using multiple variable assignment, which is a convenient way of assigning multiple variables at once. </p> 
      
      ```sh
      from sklearn.model_selection import train_test_split
      x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
      ``` 
      <p align="justify"> This line of code imports the LinearRegression class from the sklearn.linear_model module. This class provides an implementation of the linear regression model that can be used to perform regression analysis on data. The linear regression model is a mathematical model that is used to model the linear relationship between a dependent variable and one or more independent variables. </p>
      
      <p align="center">
        <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/Liner%20Regression1.png" alt="Mising Value">
        <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/Liner%20Regression2.png" alt="Mising Value">
      </p>
      
      <p align="justify"> The first program displays a scatter plot for the training data (x_train and y_train) with green dot color (color="green"). Then, the program also displays a red line (color="red") representing the trained linear regression model using the training data. The line shows the linear relationship between the independent variable (x_train) and dependent variable (y_train) in the training data. The second program displays a scatter plot for the testing data (x_test and y_test) with blue dot color (color="blue"). Then, the program also displays the same red line (color="red") as the first program, representing the trained linear regression model using the training data. The line also shows the linear relationship between the independent variable (x_train) and dependent variable (y_train), but the model is tested on unseen testing data. By plotting these graphs, we can visualize the performance of the linear regression model in predicting the testing data based on the trained training data
   
   7. logistic Regression
   
      <p align="justify"> Logistic Regression is a statistical method used for binary classification problems where the output variable takes only two values, typically 0 or 1. It is a type of generalized linear model that uses a logistic function to model the relationship between the independent variables and the probability of the binary outcome. In logistic regression, the input variables are combined linearly using weights or coefficients to predict a log-odds output. This log-odds output is then transformed using the logistic function to produce the final probability prediction. Logistic Regression is a popular classification algorithm in Machine Learning because it is simple, fast, and interpretable. It is commonly used in various applications such as fraud detection, customer churn prediction, and disease diagnosis. </p>
   
   8. KKN
      
       <p align="justify"> K Nearest Neighbors (KNN) is a Machine Learning algorithm used for classification and regression problems. Essentially, the KNN algorithm looks for the K nearest neighbors of a given data point and determines the label or regression value based on the majority of the neighbors' labels. In the case of classification, KNN determines the class label by taking the majority label from the nearest neighbors. In contrast, in the case of regression, KNN determines the regression value by taking the average value from the nearest neighbors.
       KNN works by calculating the distance between the given data point and all other data points in the dataset. The most common distance metrics used are Euclidean distance or Manhattan distance. KNN has the advantages of being easy to implement, performing well on simple data, and being easy to interpret. However, KNN is not suitable for complex or high-dimensional data, and it can also be computationally expensive to calculate distances in large datasets. </p>
        
   9. SVM
      
       <p align="justify"> SVM or Support Vector Machine is a Machine Learning algorithm used for classification and regression problems. The main objective of SVM is to find the most optimal hyperplane or boundary line to separate the two classes in the given data. SVM searches for a hyperplane by maximizing the margin, which is the closest distance between the hyperplane and the nearest data points from both classes. In addition, SVM can also use kernels to map the data into a higher-dimensional space, allowing SVM to find more complex hyperplanes for non-linearly separable data. SVM has advantages because it can work well on data with many features, whether the data is linearly separable or not. Additionally, SVM has the ability to overcome overfitting on training data and can be used for multi-class problems. However, SVM also has drawbacks, including taking longer to train the model on large and complex data, and having difficulty in handling data with significant noise and outliers.  </p>
     
   10. Decision Tree
      
       <p align="justify"> Decision Tree is one of the machine learning algorithms used in data science to build prediction or classification models based on a series of decisions made on the given data. The way the Decision Tree algorithm works is by dividing the data into several groups or nodes based on certain criteria, such as feature or attribute values, so that each node represents a smaller and more homogeneous subgroup in terms of the target variable to be predicted. Each node is then divided again into smaller and more homogeneous subgroups until a termination condition is reached. The termination condition can be a minimum sample size at each node or a sufficient level of accuracy at each node. Once the decision tree is formed, the model can be used to predict the target variable based on new input features by following the appropriate path in the decision tree. The Decision Tree algorithm can be applied to various types of problems, such as classification, regression, or clustering. The advantages of the Decision Tree algorithm are that it is easy to understand and interpret, can be used on qualitative and quantitative data, and can handle noisy or incomplete data. However, its weakness is that it tends to overfit and cannot handle complex problems well. </p>
      
   11. Random Forest
      
       <p align="justify"> Random Forest is a machine learning algorithm used in data science for classification, regression, and other types of data analysis tasks. It is an ensemble method that combines multiple decision trees to create a more accurate and robust model. The Random Forest algorithm works by creating a large number of decision trees, each using a random subset of the features and a random subset of the data. This randomness helps to reduce overfitting and improve generalization performance. When making a prediction, each decision tree in the forest produces its own prediction, and the final prediction is determined by taking the majority vote of all the decision trees. The advantages of the Random Forest algorithm include its ability to handle noisy and missing data, its resistance to overfitting, and its suitability for large datasets. However, its main disadvantage is that it can be computationally expensive and require significant resources to train and optimize the model. Random Forest has become a popular algorithm in machine learning due to its ability to provide high accuracy and robustness in a wide range of data analysis tasks. </p>
      
   12. K Means Clustering
      
       <p align="justify"> K Means Clustering is a popular unsupervised machine learning algorithm used for clustering data into groups or clusters based on their similarity. The algorithm works by partitioning data into K clusters, where K is the number of clusters specified by the user. The K Means Clustering algorithm starts by randomly selecting K data points as centroids, which are the initial cluster centers. The algorithm then assigns each data point to the nearest centroid, creating K clusters. The centroids are then updated by calculating the mean of all the data points in each cluster. The algorithm iteratively repeats these two steps of cluster assignment and centroid update until convergence, which occurs when the centroids no longer change or the maximum number of iterations is reached. The K Means Clustering algorithm has several advantages, including its simplicity, speed, and ability to handle large datasets. However, it also has some limitations, such as its sensitivity to the initial centroids and its tendency to produce spherical clusters. K Means Clustering is widely used in data science for various applications such as image segmentation, customer segmentation, and anomaly detection.</p> 
   
   13. Hyper Parameter Optimization with GridSearchCV
      
       <p align="justify"> Hyperparameter optimization is a technique used in machine learning to tune the parameters of a model for better performance. GridSearchCV is a popular method for hyperparameter optimization that exhaustively searches over a specified grid of hyperparameters to find the best combination of parameters for a given model. GridSearchCV works by creating a grid of all possible hyperparameter values and then evaluating the model with each combination of hyperparameters using cross-validation. The performance metric used to evaluate the model is typically the mean accuracy or F1 score. The hyperparameters to be tuned and their ranges are specified by the user, and the GridSearchCV algorithm then trains the model with each combination of hyperparameters in the grid. Once all the combinations have been evaluated, the algorithm returns the hyperparameters that resulted in the best performance. GridSearchCV is a powerful tool for optimizing hyperparameters and can save time and effort compared to manually tuning hyperparameters. However, it can also be computationally expensive, especially for large grids of hyperparameters or for complex models. Overall, GridSearchCV is a useful technique for improving the performance of machine learning models and is commonly used in data science for various applications such as image recognition, text classification, and time series forecasting. </p>
      
   14. Machine Learning Pipeline
      
       <p align="justify"> A machine learning pipeline is a sequence of data processing and modeling steps that are chained together to transform raw data into a final prediction model. A typical pipeline involves several stages, including data cleaning and preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation. The pipeline starts with raw data, which is then preprocessed to remove missing values, handle outliers, and transform features as needed. Next, features are engineered by selecting, transforming, and combining the most relevant features to improve model accuracy. Then, various models are selected and trained on the preprocessed data. After training, the model is optimized by tuning hyperparameters using techniques such as grid search, cross-validation, or Bayesian optimization. Finally, the performance of the model is evaluated on a holdout dataset to assess its accuracy and generalization capability. The use of machine learning pipelines helps streamline the process of building and optimizing models, making it easier to experiment with different techniques and compare their performance. It also enables automation and reproducibility of the modeling process, making it easier to iterate and refine the model over time. </p>
      
   
   15. Machine Learning Model Evaluation Metrics 
      
       <p align="justify"> Machine learning model evaluation metrics are used to measure the performance of a machine learning model. The choice of evaluation metrics depends on the problem type, such as classification or regression, and the specific requirements of the problem at hand.  Some common evaluation metrics used in machine learning include: </p>
       
       * Accuracy: measures the percentage of correct predictions out of all predictions made by the model.
       * Precision: measures the percentage of true positives out of all positive predictions made by the model. It is a measure of how precise the model's positive predictions are.
       * Recall: measures the percentage of true positives out of all actual positive cases in the data. It is a measure of how well the model can identify positive cases.
       * F1 score: is the harmonic mean of precision and recall, and provides a balance between the two metrics.
       * ROC-AUC: measures the area under the receiver operating characteristic (ROC) curve, which is a plot of true positive rate against false positive rate at different classification thresholds. It is commonly used for binary classification problems.
       * Mean Squared Error (MSE): is used for regression problems to measure the average squared difference between the predicted and actual values.
       * R-squared (R²): measures the proportion of variance in the dependent variable that is explained by the independent variables. It is also used for regression problems.
       * These are just a few examples of the many evaluation metrics that can be used in machine learning. The choice of evaluation metric should be based on the specific requirements of the problem at hand and the nature of the data.
 
<!-- Cloud Coumputing for Machine Learning -->
## ☁️ Cloud Coumputing for Machine Learning 
 
   <p align="justify"> Cloud computing is an ideal solution for machine learning as it provides a scalable, reliable, and cost-effective way to perform resource-intensive machine learning tasks. Cloud providers offer a range of services for machine learning, including managed services for training and deploying models, big data analytics tools, and computing infrastructure. Some of the benefits of using cloud computing for machine learning are:
   
   * Scalability: Cloud providers offer elastic computing resources that can scale up or down based on the demand. This allows users to easily scale their machine learning workloads as needed, without having to worry about the underlying infrastructure.
   * Cost-effectiveness: Cloud computing can help reduce the costs of running machine learning workloads. Cloud providers offer pay-as-you-go pricing models, allowing users to only pay for the resources they use. This eliminates the need to invest in expensive hardware and software upfront.
   * Flexibility: Cloud computing provides users with the flexibility to choose the right tools and services for their machine learning workloads. Cloud providers offer a range of machine learning services and tools, allowing users to pick and choose the ones that best suit their needs.
   * Security: Cloud providers have robust security measures in place to protect user data and applications. This can help users ensure the security of their machine learning workloads, without having to worry about the underlying infrastructure.
    Overall, cloud computing provides a highly scalable, cost-effective, and flexible platform for machine learning that can help organizations accelerate their machine learning initiatives and stay competitive in today's rapidly changing business landscape. </p>
   
   1. Introduction AWS
      
      <p align="justify"> AWS (Amazon Web Services) is a cloud computing platform provided by Amazon. It provides a wide range of cloud-based services, including computing power, storage, and databases, as well as tools for analytics, machine learning, and IoT (Internet of Things). AWS offers a highly scalable and reliable cloud infrastructure that enables businesses to run their applications and services with ease. WS offers a wide range of services, including: </p>
      
      * mputing: AWS offers a range of computing options, including EC2 (Elastic Compute Cloud), Lambda, and ECS (Elastic Container Service).
      * Storage: AWS offers a range of storage options, including S3 (Simple Storage Service), EBS (Elastic Block Store), and Glacier.
      * Databases: AWS offers a range of database options, including RDS (Relational Database Service), DynamoDB, and Neptune.
      * Analytics: AWS offers a range of analytics options, including EMR (Elastic MapReduce), Kinesis, and QuickSight.
      * Machine Learning: AWS offers a range of machine learning options, including SageMaker, Rekognition, and Comprehend.
      * IoT: AWS offers a range of IoT options, including IoT Core, Greengrass, and IoT Analytics.
      
      <p align="justify"> AWS also provides a range of management and security tools to help users manage their resources and ensure the security of their applications and data. These include AWS CloudFormation, AWS CloudTrail, and AWS Identity and Access Management (IAM). Overall, AWS provides a comprehensive cloud computing platform that enables businesses to run their applications and services with ease, while also providing a range of tools and services for analytics, machine learning, and IoT. </p>
     
   2. Different AWS Services
      
      <p align="justify"> Amazon S3 (Simple Storage Service) is a cloud storage service provided by Amazon Web Services (AWS). S3 enables users to store, retrieve, and manage large amounts of data and files quickly, securely, and easily. The service consists of buckets to store data, and each bucket can store a large number of objects, such as images, videos, documents, and more. S3 has several features, such as: </p>
     
      * High durability and availability - S3 redundantly stores each object in multiple availability zones across the AWS region, ensuring high availability and minimizing the risk of data loss.
      * Scalability - S3 can easily handle storing data from a few bytes to petabytes or more.
      * Security - S3 provides strong access controls and data encryption to protect data from unauthorized access.
      * Flexibility - S3 can be used for various use cases, including file storage, data backup and archiving, hosting static websites, and more.
      * The service also has a usage-based pricing model, where users only pay for the storage capacity and data transfer they use.
      
   
   3. Introduction to AWS SageMaker
      
      <p align="justify"> Amazon SageMaker is a machine learning service provided by Amazon Web Services (AWS) that allows developers and data scientists to build, train, and deploy machine learning models quickly and easily. SageMaker provides a fully-managed platform that enables users to eliminate the heavy lifting typically associated with building, training, and deploying machine learning models. With SageMaker, users can easily create and train machine learning models using popular frameworks like TensorFlow and PyTorch, without needing to worry about the underlying infrastructure.
      SageMaker also provides pre-built algorithms for common use cases, making it easy to get started with machine learning. In addition, SageMaker provides tools for data labeling, model tuning, and model hosting, making it easy to deploy machine learning models at scale. Users can deploy models as web services that can be easily integrated into applications and can handle millions of requests per day. SageMaker also provides features for model explainability and monitoring, making it easy to understand how the model is making predictions and to detect any issues that may arise over time.
      Overall, SageMaker simplifies the machine learning workflow and makes it easy for developers and data scientists to build and deploy machine learning models quickly and efficiently. </p>
      

   4. First Machine Learning Practical on AWS SageMaker
     
       <p align="justify"> Great! If you're looking to get started with machine learning on AWS SageMaker, here are some steps to follow:
      
      * Create an AWS account: If you haven't already, create an AWS account by visiting https://aws.amazon.com and following the instructions provided.
      * Sign in to AWS SageMaker: Once you have an AWS account, navigate to the AWS Management Console and sign in to AWS SageMaker.
      * Create a notebook instance: In AWS SageMaker, create a new notebook instance by clicking on "Notebook instances" in the left-hand menu and then clicking the "Create notebook instance" button.
      * Choose an instance type: Select an instance type for your notebook instance. This determines the computing resources that will be available to you for running machine learning models.
      * Create a new notebook: Once your notebook instance is ready, create a new notebook and give it a name.
      * Write and run code: You can now start writing and running Python code in your notebook using popular machine learning libraries like TensorFlow, PyTorch, or scikit-learn. SageMaker also provides a number of built-in algorithms and pre-built notebooks to help you get started.
      * Monitor and debug: As you run your machine learning models, you can use SageMaker's monitoring and debugging tools to identify and fix any issues that arise.
   These are just some basic steps to get started with machine learning on AWS SageMaker. There are many additional features and capabilities to explore as you become more familiar with the platform
      </p>
   
   5. Built in ML Algorithms in AWS SageMaker
      
      <p align="justify"> AWS SageMaker provides a wide range of built-in machine learning algorithms that can be used to train your own models on your data. Here are some of the built-in algorithms available in SageMaker:
      
      * Linear Learner: A binary and multiclass classification algorithm that can also be used for regression tasks.
      * XGBoost: A popular gradient boosting algorithm for classification and regression tasks.
      * DeepAR: A neural network-based algorithm for time series forecasting.
      * K-Means: An unsupervised learning algorithm used for clustering.
      * Image Classification: A computer vision algorithm that uses deep learning to classify images.
      * Object Detection: A computer vision algorithm that uses deep learning to detect objects in images.
      * Semantic Segmentation: A computer vision algorithm that uses deep learning to segment images into different categories.
      * Factorization Machines: A supervised learning algorithm used for recommendation systems.
      * Random Cut Forest: An unsupervised learning algorithm used for anomaly detection.
      * Principal Component Analysis (PCA): A dimensionality reduction algorithm used for feature extraction and data compression.

      These built-in algorithms in AWS SageMaker are pre-optimized for high-performance and scalable machine learning, and can be easily trained and deployed on large datasets. You can also customize these algorithms using SageMaker's built-in frameworks like TensorFlow and PyTorch, or use your own custom algorithms with SageMaker's Bring Your Own Algorithm (BYOA) feature. </p>
      
   6. Linear Learner Algorithm Practical Implementation
      
      <p align="justify"> Linear Learner Algorithm is a supervised learning algorithm used for regression and binary classification tasks in Amazon SageMaker. In practical implementation, you can use Linear Learner to train a model on your dataset, tune the model parameters, and evaluate its performance. Here are the basic steps for practical implementation of the Linear Learner algorithm in Amazon SageMaker:
      * Prepare your data: You need to prepare your dataset in a format that can be used by the Linear Learner algorithm. This involves cleaning the data, splitting it into training and testing sets, and formatting it in the appropriate input format for SageMaker.
      * Train the model: Once you have prepared the data, you can use the Linear Learner algorithm to train the model. You need to specify the algorithm hyperparameters, such as the learning rate, regularization parameter, and number of epochs.
      * Evaluate the model: After training the model, you need to evaluate its performance on a test dataset. You can use metrics such as accuracy, precision, recall, and F1 score to assess the model performance.
      * Deploy the model: Once you are satisfied with the model performance, you can deploy it to a SageMaker endpoint. This allows you to use the model to make predictions on new data.
      * Monitor and iterate: Finally, you need to monitor the model performance and iterate on the model as needed. This involves tuning the hyperparameters, adding new features to the dataset, or retraining the model on additional data. </p>
      
   7. No Code ML Using AWS SageMaker Canvas
      
      <p align="justify"> AWS SageMaker Canvas is a visual interface that allows you to build and deploy machine learning models without writing any code. It provides a drag-and-drop interface that enables you to build a machine learning workflow by selecting and configuring pre-built components. With SageMaker Canvas, you can easily create a machine learning pipeline that includes data preprocessing, feature engineering, model training, and deployment. The interface provides pre-built components for each step in the pipeline, so you can simply drag and drop the components and configure their parameters. Here are the basic steps for building a no-code machine learning model using SageMaker Canvas: 
   
      * Import and explore your data: You can import your data into SageMaker Studio and explore it using various data visualization tools. You can also use SageMaker Data Wrangler to clean, transform, and preprocess your data.
      * Build your ML workflow: Once you have explored and preprocessed your data, you can start building your machine learning workflow using SageMaker Canvas. You can select the pre-built components for data preprocessing, feature engineering, model training, and deployment and connect them to build a workflow.
      * Configure the components: You can configure the parameters of each component in the workflow, such as the type of data transformation, the hyperparameters of the model, and the type of deployment.
      * Run and monitor the workflow: After configuring the components, you can run the workflow and monitor its progress using various tools provided by SageMaker, such as logs and metrics.
      * Deploy the model: Once the model is trained, you can deploy it to a SageMaker endpoint and use it to make predictions on new data.
      * No-code machine learning using SageMaker Canvas enables users with limited programming experience to build and deploy machine learning models quickly and easily. </p>
   
   8. AWS SageMaker MarketPlace
      
      <p align="justify"> AWS SageMaker Marketplace is a platform where developers, data scientists, and machine learning enthusiasts can discover, buy, and sell machine learning algorithms, models, and other related resources. It provides a one-stop-shop for users to find pre-built machine learning models, frameworks, and tools, which can be easily integrated into their own workflows. The marketplace offers a wide range of pre-built solutions from various providers, including AWS and third-party vendors. The solutions cover a variety of machine learning use cases, such as computer vision, natural language processing, fraud detection, and many more. Using the AWS SageMaker Marketplace, users can quickly deploy and integrate pre-built models into their own workflows, without needing to build the models from scratch. This saves users a lot of time and resources, and enables them to focus on solving their specific business problems.
      The marketplace provides an easy-to-use interface for browsing and selecting pre-built models and algorithms. Users can search for solutions based on their specific use case, filter by price or rating, and view product details and reviews from other users. Once a user finds a solution they like, they can quickly deploy it to their SageMaker environment with just a few clicks. Overall, the AWS SageMaker Marketplace is a valuable resource for developers and data scientists who want to accelerate their machine learning projects by leveraging pre-built models and algorithms. </p>
      
<!-- Deep Learning -->
## 🧪 Deep Learning

<p align="justify"> Deep learning is a subset of machine learning that involves building and training artificial neural networks to recognize patterns in data. It is a type of artificial intelligence that allows machines to learn from large amounts of data, and make predictions or decisions based on that learning. Deep learning models are designed to simulate the human brain, with layers of artificial neurons that process input data and make predictions or decisions based on that data. These models are trained on large datasets using techniques such as gradient descent, backpropagation, and stochastic gradient descent.
Deep learning has revolutionized many fields, such as computer vision, natural language processing, and speech recognition. It has enabled machines to achieve state-of-the-art performance on tasks such as object detection, image classification, language translation, and voice recognition. Some popular deep learning frameworks and tools include TensorFlow, PyTorch, Keras, and Caffe. These frameworks provide a range of tools for building and training deep learning models, as well as pre-trained models that can be used for various tasks.
Overall, deep learning has enabled machines to learn and make decisions based on complex data, and has the potential to transform many industries and fields in the coming years.</p>

   1. Artificial Neural Network(ANN)
      
      <p align="justify"> Artificial Neural Network (ANN) is a type of deep learning model that is inspired by the structure and function of the human brain. It is composed of layers of interconnected artificial neurons, which process information and make predictions or decisions based on that information. An artificial neuron in an ANN takes in input from other neurons or from external sources, and applies a mathematical function to that input to generate an output. This output is then passed on to other neurons in the next layer, and so on, until the final layer produces the model's output. ANNs are trained on large datasets using techniques such as backpropagation and gradient descent. During training, the weights of the connections between neurons are adjusted to minimize the difference between the model's predictions and the actual outputs.
      ANNs can be used for a wide range of tasks, such as image classification, speech recognition, natural language processing, and many more. They have enabled machines to achieve state-of-the-art performance on many tasks, and have the potential to transform many industries and fields. Some popular deep learning frameworks, such as TensorFlow and PyTorch, provide tools for building and training ANNs, as well as pre-trained models that can be used for various tasks. Overall, ANNs are a powerful tool in the field of deep learning and have the potential to revolutionize many fields in the coming years. </p>
       
   2. Activation Functions Neural Networks 
      
      <p align="justify"> Activation functions are a critical component of artificial neural networks (ANNs). They introduce nonlinearity into the output of artificial neurons, allowing ANNs to learn and model complex, nonlinear relationships in data. An activation function takes in the weighted sum of the inputs to an artificial neuron and applies a mathematical function to that sum to produce the neuron's output. The output is then passed on to other neurons in the network.
      There are many types of activation functions used in ANNs, including:
         
         * Sigmoid: The sigmoid function produces an S-shaped curve that maps any input to a value between 0 and 1. It is commonly used in the output layer of binary classification tasks.
         * ReLU (Rectified Linear Unit): ReLU applies a rectifier function to the input, outputting the input if it is positive and 0 if it is negative. It is widely used in hidden layers of deep neural networks.
         * Tanh: The hyperbolic tangent function is similar to the sigmoid function but maps inputs to a value between -1 and 1. It is often used in the hidden layers of neural networks.
         * Softmax: The softmax function is used in the output layer of multi-class classification tasks. It maps the input to a probability distribution over the output classes.
         * Leaky ReLU: Leaky ReLU is similar to ReLU but allows a small, non-zero gradient when the input is negative. This can help with the vanishing gradient problem in deep neural networks.
   
      Overall, the choice of activation function can have a significant impact on the performance of an artificial neural network. By introducing nonlinearity, activation functions allow ANNs to model complex relationships in data, and different activation functions may be more suitable for different types of tasks. </p>
      
   3. Optimizers in Neural Networks 
      
      <p align="justify"> Optimizers are a key component of training artificial neural networks (ANNs). They are algorithms that adjust the weights of the connections between neurons during training to minimize the difference between the model's predictions and the actual outputs.There are many different optimizer algorithms that can be used in ANNs, each with its own strengths and weaknesses. Some popular optimizers include:
         
         * Stochastic Gradient Descent (SGD): SGD is a simple and widely used optimizer that updates the weights of the connections between neurons based on the gradient of the loss function with respect to the weights. It works well for smaller datasets, but can be slow and can get stuck in local minima.
         * Adam: Adam is an adaptive optimizer that adjusts the learning rate during training based on the first and second moments of the gradients. It works well for large datasets and can converge quickly, but can be sensitive to the choice of hyperparameters.
         * Adagrad: Adagrad is an adaptive optimizer that adjusts the learning rate for each weight based on the history of the gradients for that weight. It works well for sparse datasets, but can converge too quickly and stop learning.
         * RMSprop: RMSprop is an adaptive optimizer that adjusts the learning rate based on the root mean square of the gradients. It works well for non-stationary problems, but can have trouble with saddle points.
      There are many other optimizer algorithms as well, each with its own tradeoffs. The choice of optimizer can have a significant impact on the performance of an ANN, and choosing the right optimizer for a particular task is an important part of the training process.  </p>
   
   4. Convolutional Neural Network(CNN)
      
      <p align="justify"> Convolutional Neural Networks (CNNs) are a type of artificial neural network that are commonly used for image and video analysis tasks. They are inspired by the organization of the visual cortex in animals, which uses a hierarchical arrangement of cells to process visual information. A CNN consists of multiple layers, including convolutional layers, pooling layers, and fully connected layers. The convolutional layers apply a set of filters to the input image, extracting features such as edges and corners. The pooling layers downsample the output of the convolutional layers, reducing the dimensionality of the feature maps. The fully connected layers take the flattened output of the pooling layers and produce the final output of the network.
      CNNs are trained using a process called backpropagation, which adjusts the weights of the connections between neurons to minimize the difference between the model's predictions and the actual outputs. The weights in the convolutional layers are learned automatically during training, allowing the network to adapt to different types of images and visual features. CNNs have achieved state-of-the-art performance on many computer vision tasks, such as image classification, object detection, and semantic segmentation. They have been used in a wide range of applications, from self-driving cars to medical imaging. Many popular deep learning frameworks, such as TensorFlow and PyTorch, provide tools for building and training CNNs.</p>
   
   5. Recurrent Neural Network(RNNs)
       
       <p align="justify"> Recurrent Neural Networks (RNNs) are a type of artificial neural network that are commonly used for processing sequential data, such as time-series data or natural language text. Unlike feedforward neural networks, which process inputs one at a time without any memory of past inputs, RNNs maintain an internal state or "memory" that allows them to process sequences of inputs. At each time step, an RNN takes in an input and produces an output and an updated internal state. The output is typically fed back into the network as input for the next time step, along with the next input in the sequence. This allows the network to maintain a memory of past inputs and produce outputs that depend on the entire input sequence, not just the current input. RNNs can be trained using a process called backpropagation through time, which is a variant of backpropagation that takes into account the temporal dependencies between the inputs and outputs. However, RNNs can suffer from the vanishing gradient problem, where the gradients used to update the network weights become very small and cause the network to stop learning.
      To address this problem, several variants of RNNs have been developed, such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), which use specialized memory cells and gating mechanisms to control the flow of information through the network and avoid the vanishing gradient problem. RNNs have been successfully applied in many areas, such as speech recognition, machine translation, and image captioning. They are also commonly used as a building block for more complex models, such as encoder-decoder architectures and attention mechanisms. Many popular deep learning frameworks, such as TensorFlow and PyTorch, provide tools for building and training RNNs and their variants. </p>
      

<!-- Project -->
## 🦾 Project

- [x] Diabetes Prediction
- [x] Medical Insurance Cost Prediction
- [ ] Gold Price Prediction Using ANN
- [ ] Implementation of CNN Using Keras and TensorFlow
- [ ] Stock Price Prediction Using LSTM 

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## 🚶 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## 📋 License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- CONTACT -->
## ☎️ Contact

Your Name -  <!-- [@your_twitter](https://twitter.com/your_username) - email@example.com -->

Project Link: <!-- [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name) -->



