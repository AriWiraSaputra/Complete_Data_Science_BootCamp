# Complete Data Science BootCamp
<p align="center">
   <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/Data%20Science.png" alt="Data Science" width="1000" height="500">
</p>

## Course overview and what will you lear?

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
## About The Project

The Complete Data Science Bootcamp project is an intensive training program designed to equip participants with skills and knowledge in the field of data science. In this project, participants will learn basic data science concepts, including data processing, statistical analysis, and data modeling. In addition, participants will also minimize the use of the latest technologies and tools in data science, such as Python, SQL, machine learning, and data visualization. The Complete Data Science Bootcamp project aims to prepare participants to become professional data scientists who are reliable and ready to compete in an increasingly competitive job market.

<!-- Numpy -->
## Numpy

NumPy (Numerical Python) is a Python library for scientific computing that provides a powerful array processing capability. NumPy provides an efficient implementation of multi-dimensional arrays, which are the main data structure for numerical computing in Python. using NumPy enables users to process numeric data efficiently and effectively, as well as accelerate the development of code and applications that require processing of numeric data.

1. Numpy Arrays
   * Zero D Arrays
   
      Zero-dimensional arrays, also known as scalar arrays, are arrays with no axes or dimensions. They are used to represent a single scalar value, such as a constant       or a single data point.
       ```sh
       import numpy as np
       a = np.array(42)
       print(a) #42
       b = np.array([1,2,3,4])
       print(b.ndim) # 1 --> ndim is used to count the number of arrays
       ```
   * 2D Arrays
   
     2D arrays, also known as matrices, are arrays with two axes or dimensions. They are used to represent tabular data or a matrix of values, such as an image with height and width. In NumPy, 2D arrays can be created using the np.array() function and passing a list of lists as the input.
     ```sh
     import numpy as np
     a = np.array([[1,2,3],[4,5,6]])
     print(a) #[[1,2,3],
              #[4,5,6]]
     print(a.ndim) #2 --> ndim is used to count the number of arrays
     ```
   * 3D Arrays
   
     Three-dimensional (3D) arrays, also known as a cube of data, are arrays with three axes or dimensions. They are used to represent data that can be thought of as a cube or a rectangular prism, such as RGB images or volume data. Each element in a 3D array is accessed using three indices, corresponding to the three axes of the cube.
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
   
      "Shape" in NumPy refers to the size or dimensions of an array. For one-dimensional arrays, the shape only contains a single integer representing the length of the array. For multi-dimensional arrays, the shape contains a tuple of integers representing the size of the array in each dimension. For example, a shape of (2, 3) would indicate a two-dimensional array with 2 elements in the first dimension and 3 elements in the second dimension.
       ```sh
       import numpy as np
       a = np.array([1,2,3,4,5])
       print(a.shape) #(5,)
       b = np.array([1,2,3,4,5],[[6,7,8,9,10]])
       print(b.shape) #(2,5) containing the elements 1 through 10 arranged in 2 rows and 5 columns
       ```
    * Reshape
    
      Reshape is a method in NumPy for changing the dimensions or shape of an array without changing its data. By using the reshape method, we can convert a one-dimensional array to a multi-dimensional array, or reshape a multi-dimensional array to a different shape with the same number of elements. For example, we can reshape a one-dimensional array with 12 elements to a two-dimensional array with a size of 3x4, or a two-dimensional array with a size of 2x6 to a three-dimensional array with a size of 3x2x2.
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

   Array indexing is a way to access specific elements or a subset of elements from a NumPy array. In NumPy, indexing starts at 0, which means that the first element in an array has an index of 0, the second element has an index of 1, and so on.
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

    Array iteration is the process of finding each element of an array in sequence to perform a specific operation. In NumPy, we can perform array iteration using a for loop. Each element of the array can be accessed using array indexing. For example, for a two-dimensional array, we can use two loops to access each element. Then, we can perform any operation on each element as needed.
    ```sh
    import numpy as np
       a = np.array([1,2,3,4,5])
       for i in a:
          print(i)
                  #1
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
               print(j) 
                  #1
                  #2
                  #3
                  #4
                  #5
                  #6
    ```
6. Silicing

    Slicing arrays is a technique for taking a portion or subset of a larger array by selecting only certain elements within that array. In NumPy, slicing is done using square brackets [ ] and the range of array indices that we want to take. For example, to take a portion of an array from index 3 up to index 6, we can use the syntax [3:7]. Additionally, we can also take elements with a certain step size using the syntax [start: end: step]. For instance, to take every second element from an array, we can use the syntax [::2].
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

   Searching and sorting arrays refer to finding specific values or ordering the elements of an array in a specific manner. In NumPy, we can use various methods to search for elements in an array, such as np.where() and np.searchsorted(). We can also sort the elements of an array in ascending or descending order using np.sort() and np.argsort(). Additionally, we can sort the elements of an array based on a specific criterion using np.lexsort() and np.partition().
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
## Pandas
Pandas is one of the Python libraries used for data analysis and data manipulation. This library provides flexible and efficient data structures for working with structured data such as SQL tables or Excel spreadsheets.

Pandas allows users to perform many tasks, such as reading and writing data in various formats, such as CSV, Excel, SQL, and others. Additionally, pandas enables users to transform data into forms suitable for analysis, such as filtering, grouping, and aggregating data. This library also has the ability to perform attractive data visualization.

Pandas is highly popular among data scientists, computer scientists, and engineers because it enables them to quickly and efficiently retrieve, manipulate, and analyze data. Additionally, pandas has excellent documentation and an active community, making it easy to find help and solutions when encountering problems or difficulties in using this library.

1. Series in Pandas

   Pandas Series is a one-dimensional data structure similar to an array but with user-defined index capabilities. Pandas Series can contain numeric data, string data, Python objects, and other data. A set of Pandas Series can also form a Pandas DataFrame. Pandas Series can be created from a Python list, NumPy array, or dictionary. In addition, we can perform various operations on Pandas Series, such as filtering, grouping, aggregation, and data visualization.
   Pandas Series are very useful in data analysis and make data processing easier for various types of problems such as machine learning, data science, and business data analysis. Due to its index capabilities and comprehensive functionality, Pandas Series has become one of the most popular data structures in the Pandas library.
   ```sh
   import pandas as pd
   a =[10,20,30,40,50]
   b = pd.Series(a)
   print(b) 
          #0  10
          #1  20
          #2  30
          #3  40
          #4  50
          dtype = int64
   c = {"one" :10,"second":100,"third":1000}
   d = pd.Series(c)
   print(d)
          #one      10
          #second  100
          #third  1000
    ```
2. Pandas DataFrame

   Pandas DataFrame is a two-dimensional data structure that is similar to SQL tables or Excel spreadsheets. Pandas DataFrame consists of rows and columns, with each column being able to have a different data type, such as numeric, string, or other Python objects. Each column in the DataFrame can be named, and the DataFrame can have user-defined indices.
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

   Read CSV is a function from the Pandas library in Python that is used to read files with CSV (Comma-Separated Values) format and convert them into a Pandas DataFrame. CSV file is a commonly used file format to store tabular data, such as data in an Excel spreadsheet. In a CSV file, each row represents a record, and each column contains data for different attributes in that record. The "read_csv" function allows us to read CSV files, load the data into a DataFrame, and perform various data manipulations on the file.
   ```sh
      import pandas as pd
      df = pd.read_csv('airport.csv')
      print(df)
   ```
   <p align="center">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/csv.png" alt="Description_image" >
   </p>

4. Analyzing DataFrame in Pandas

   Analyzing DataFrame in Pandas involves performing various operations on the DataFrame to gain insights into the data, identify patterns, and make data-driven decisions. Some of the common operations that can be performed on a Pandas DataFrame for analysis are:
   
   * Data Exploration - exploring the data to get an idea of its structure, quality, and distribution using various Pandas functions like head(), tail(), info(), describe(), etc.
   * Data Cleaning - cleaning the data by handling missing or null values, removing duplicates, correcting data types, etc.
   * Data Transformation - transforming the data to create new variables, combine variables, apply functions, filter data, etc.
   * Data Aggregation - aggregating the data to create summary statistics, group data by categories, pivot tables, etc.
   * Data Visualization - visualizing the data using various charts, plots, and graphs to identify patterns and relationships in the data.

   By performing these operations, we can gain a better understanding of the data and use it to make data-driven decisions. Pandas provides a rich set of functions and tools to perform these operations and analyze data efficiently.
   
   
   | Image 1 =  head() - This function is used to display the first five rows of the DataFrame| Image 2 = tail() - This function is used to display the last five rows of the DataFrame |
   | ------------- | ------------- |
   | <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/head.PNG.png" alt="Head" width="800" height="300">| <img src= "https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/tail.PNG.png" width="800" height="300"> |
   | Image 3 info() - This function is used to display general information about the DataFrame, such as the number of rows and columns, data types, and non-null values | Image 4 describe() - This function is used to display statistical summary of the DataFrame, such as mean, median, minimum and maximum values, etc |
   | <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/info.PNG.png" width="800" height="300"> | <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/describe.PNG.png" width="800" height="300"> |


<!-- Data Visualization -->
## Data Visualization
1. Introduction to Matplotlib
   
   Matplotlib is a Python library used for creating visualizations, such as graphs, charts, and plots. It provides a flexible and powerful platform for data visualization, and is widely used in data science, scientific research, and engineering fields.

   Matplotlib allows users to create a variety of visualization types, including line plots, scatter plots, bar plots, histograms, and more. It also provides customization options for controlling aspects such as color, fonts, and sizes.
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

   Seaborn is a Python data visualization library based on Matplotlib that provides a high-level interface for creating informative and attractive statistical graphics. Seaborn makes it easy to create a wide range of statistical visualizations, including scatter plots, line plots, bar plots, histograms, heatmaps, and more. Seaborn has several advantages over Matplotlib, including more attractive default styles, simpler syntax for common plot types, and built-in functionality for working with complex datasets. Seaborn also includes advanced features like built-in support for visualizing distributions, regression models, and statistical relationships between variables. Overall, Seaborn is a powerful tool for data scientists and analysts who need to quickly create visualizations that help them understand and communicate complex data
   
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
## Data Preprocessing
1. Handling Missing Values

   Handling missing values refers to the process of identifying and dealing with the absence of data in a dataset. Missing values can occur for various reasons, such as data entry errors, equipment failure, or participant dropout. Handling missing values is an important step in data preprocessing because most machine learning algorithms cannot handle missing values.
   exemple : 
   ```sh
   import pandas as pd
   df = pd.read_csv('Placement_Data_Full_Class.csv')
   df()
   ```
   <p align="center">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/datasheet%20titanic.PNG" alt="Mising Value">
   </p>
   
   There are missing values in the salary column of the analyzed data. This indicates that there are employees whose salary information has not been inputted or incomplete data. Before conducting further analysis, it is necessary to fill in the missing values with an appropriate method such as using the mean, median, or mode, to ensure accurate and reliable analysis results.
   ```sh
   df.isnull().sum()
   ```
   df.isnull().sum() is a code in Python used to count the number of missing or null values in each column of a DataFrame df. isnull() is a method used to mark each value in the DataFrame as either True if it is null, or False if it is not null. Then, sum() is used to count the number of True values for each column. Therefore, df.isnull().sum() will return a Series with the column index of the DataFrame df, and the values as the count of missing values in each column. This is very useful in data exploration to ensure that there are no missing data in certain columns of the DataFrame.
   <p align="center">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/datasheet%20titanic.PNG" alt="Mising Value">
   </p>
   
   ```sh
   df['salary'] = df['salary'].fillna(df['salary'].mode()[0])
   ```
   This line of code is filling missing values in the 'salary' column of a pandas DataFrame with the mode (most frequent value) of that column. The fillna() method of a pandas DataFrame is used to replace missing (NaN) values with a specified value. In this case, the specified value is the mode of the 'salary' column, which is obtained by calling the mode() method on the 'salary' column. The [0] index is used to extract the first (and only) value from the mode result, since the mode method can return multiple modes if there are ties.
   
   This code assumes that the 'salary' column has missing values and that replacing them with the mode is an appropriate strategy for handling missing data in that column. It's important to note that filling missing values with the mode can introduce bias in the data if the mode is not representative of the underlying distribution. It's always a good practice to carefully consider different strategies for handling missing data and their potential impact on downstream analysis.
   
   ```sh
   df.isnull().sum()
   ```
   <p align="center">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/datasheet%20titanic.PNG" alt="Mising Value">
   </p>
   
   ```sh
   df.isnull().sum()
   ```
   <p align="center">
      <img src="https://github.com/AriWiraSaputra/Complete_Data_Science_BootCamp/blob/main/datasheet%20titanic.PNG" alt="Mising Value">
   </p> 
   
    All 'salary' values have been filled and there are no more null values. The 'salary' column has been updated to fill all previously missing values. Null values in the 'salary' column have been replaced with the mode, resulting in a complete dataset with no missing values for 'salary'.
    
2. Feature Enconding
3. Feature Scealing











<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

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
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
