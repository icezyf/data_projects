{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "[DS] Spark Movie Recommendation.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyOEwpF05TuMGA5vdvVDD9PY"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AJELaTdhdC2I"
      },
      "source": [
        "# **Movie Recommendation in Spark** #\n",
        "In this notebook, an Alternating Least Squares (ALS) algorithm with Spark APIs would be applied to predict the ratings for the movies in [MovieLens Small Dataset](https://grouplens.org/datasets/movielens/latest/)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PFNThdrFdyvh"
      },
      "source": [
        "*Analyzed by Crystella Yufei Zheng*"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gg2JAn95edUc"
      },
      "source": [
        "### **0. Data ETL and Data Exploration** ###"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ezxdfJU_dtWl"
      },
      "source": [
        "!apt-get install openjdk-8-jdk-headless -qq > /dev/null\n",
        "!wget -q https://www-us.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz\n",
        "!tar xf spark-3.1.2-bin-hadoop3.2.tgz\n",
        "!pip install -q findspark"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_hOY87x6fIX4"
      },
      "source": [
        "# Miscellaneous operating system interfaces\n",
        "import os \n",
        "# os.environ: A mapping object representing the string environment. \n",
        "# behaves like a python dictionary, so all the common dictionary operations like get \n",
        "# and set can be performed. \n",
        "os.environ[\"JAVA_HOME\"] = \"/usr/lib/jvm/java-8-openjdk-amd64\"\n",
        "os.environ[\"SPARK_HOME\"] = \"/content/spark-3.1.2-bin-hadoop3.2\""
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wq0TiM8vfmRS"
      },
      "source": [
        "import findspark\n",
        "# Provides findspark.init() to make pyspark importable as a regular library.\n",
        "findspark.init() \n",
        "from pyspark.sql import SparkSession\n",
        "spark = SparkSession.builder.master(\"local[*]\").getOrCreate()"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mWFhhCH0fpBP",
        "outputId": "bee83c27-f442-4a09-c4d6-aff0e2dcb825"
      },
      "source": [
        "!ls"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "sample_data  spark-3.1.2-bin-hadoop3.2\tspark-3.1.2-bin-hadoop3.2.tgz\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "tsSGoLysfs9_",
        "outputId": "68894b54-dbcf-41e8-9d84-652665e98b82"
      },
      "source": [
        "spark.version"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "'3.1.2'"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tEiA2ToGfvXs"
      },
      "source": [
        "from csv import reader\n",
        "from pyspark.sql import Row \n",
        "from pyspark.sql.types import *\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "import warnings\n",
        " \n",
        "os.environ[\"PYSPARK_PYTHON\"] = \"python3\""
      ],
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7hTb302Df0gI",
        "outputId": "e148aa6e-4dd4-4b75-bfc3-1276f21fa6e4"
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Y9vkQl8RgTA2",
        "outputId": "6faec096-4979-415b-8e75-7c8fb953f153"
      },
      "source": [
        "!ls"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "drive  sample_data  spark-3.1.2-bin-hadoop3.2  spark-3.1.2-bin-hadoop3.2.tgz\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zVM3MHkogC3p"
      },
      "source": [
        "link_df = spark.read.load(\"/content/drive/MyDrive/Colab Notebooks/Data/ml-latest-small/links.csv\", format='csv', header = True)\n",
        "movie_df = spark.read.load(\"/content/drive/MyDrive/Colab Notebooks/Data/ml-latest-small/movies.csv\", format='csv', header = True)\n",
        "rating_df = spark.read.load(\"/content/drive/MyDrive/Colab Notebooks/Data/ml-latest-small/ratings.csv\", format='csv', header = True)\n",
        "tag_df = spark.read.load(\"/content/drive/MyDrive/Colab Notebooks/Data/ml-latest-small/tags.csv\", format='csv', header = True)"
      ],
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EU9HV6TzgMqp",
        "outputId": "44500801-1b65-4cec-a228-3ec330cb921a"
      },
      "source": [
        "movie_df.show(5)"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+-------+--------------------+--------------------+\n",
            "|movieId|               title|              genres|\n",
            "+-------+--------------------+--------------------+\n",
            "|      1|    Toy Story (1995)|Adventure|Animati...|\n",
            "|      2|      Jumanji (1995)|Adventure|Childre...|\n",
            "|      3|Grumpier Old Men ...|      Comedy|Romance|\n",
            "|      4|Waiting to Exhale...|Comedy|Drama|Romance|\n",
            "|      5|Father of the Bri...|              Comedy|\n",
            "+-------+--------------------+--------------------+\n",
            "only showing top 5 rows\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RAzdocY_hRaT",
        "outputId": "ab3f1afa-4a2d-4354-cc73-11b23289f702"
      },
      "source": [
        "rating_df.show(5)"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+------+-------+------+---------+\n",
            "|userId|movieId|rating|timestamp|\n",
            "+------+-------+------+---------+\n",
            "|     1|      1|   4.0|964982703|\n",
            "|     1|      3|   4.0|964981247|\n",
            "|     1|      6|   4.0|964982224|\n",
            "|     1|     47|   5.0|964983815|\n",
            "|     1|     50|   5.0|964982931|\n",
            "+------+-------+------+---------+\n",
            "only showing top 5 rows\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WWly0S95hUzp",
        "outputId": "40024b59-0f44-42e9-e0ef-21313e85ded2"
      },
      "source": [
        "link_df.show(5)"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+-------+-------+------+\n",
            "|movieId| imdbId|tmdbId|\n",
            "+-------+-------+------+\n",
            "|      1|0114709|   862|\n",
            "|      2|0113497|  8844|\n",
            "|      3|0113228| 15602|\n",
            "|      4|0114885| 31357|\n",
            "|      5|0113041| 11862|\n",
            "+-------+-------+------+\n",
            "only showing top 5 rows\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fwjUhHLIhXiB",
        "outputId": "eb3ee33d-4d65-46e9-e0e8-f66a038bded7"
      },
      "source": [
        "tag_df.show(5)"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+------+-------+---------------+----------+\n",
            "|userId|movieId|            tag| timestamp|\n",
            "+------+-------+---------------+----------+\n",
            "|     2|  60756|          funny|1445714994|\n",
            "|     2|  60756|Highly quotable|1445714996|\n",
            "|     2|  60756|   will ferrell|1445714992|\n",
            "|     2|  89774|   Boxing story|1445715207|\n",
            "|     2|  89774|            MMA|1445715200|\n",
            "+------+-------+---------------+----------+\n",
            "only showing top 5 rows\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-Pb4UAIZhZJl",
        "outputId": "74ef15a3-841f-41f8-951f-fa78394f8ef1"
      },
      "source": [
        "tmp1 = rating_df.groupBy(\"userId\").count().toPandas()['count'].min()\n",
        "tmp2 = rating_df.groupBy(\"movieId\").count().toPandas()['count'].min()\n",
        "tmp3 = rating_df.groupBy(\"movieId\").count().where('count=1').count()\n",
        "tmp4 = rating_df.select('movieId').distinct().count()\n",
        "\n",
        "print(f'''For the users that rated movies and the movies that were rated:\n",
        "Minimum number of ratings per user is {tmp1}.\n",
        "Minimum number of ratings per movie is {tmp2}.\n",
        "{tmp3} out of {tmp4} movies are rated by only one user.''')"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "For the users that rated movies and the movies that were rated:\n",
            "Minimum number of ratings per user is 20.\n",
            "Minimum number of ratings per movie is 1.\n",
            "3446 out of 9724 movies are rated by only one user.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gC4RpZThhzES",
        "outputId": "df8a9c35-6b2c-4fc2-8c90-45c40783ba66"
      },
      "source": [
        "# change the type of rating into float, the type of userID and movieId into interger\n",
        "rating_pd = rating_df.toPandas()\n",
        "rating_pd['movieId'] = rating_pd['movieId'].astype('int64')\n",
        "rating_pd['userId'] = rating_pd['userId'].astype('int64')\n",
        "rating_pd['rating'] = rating_pd['rating'].astype('float64')\n",
        "rating_pd = rating_pd.drop(['timestamp'], axis=1)\n",
        "rating_pd.info()"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 100836 entries, 0 to 100835\n",
            "Data columns (total 3 columns):\n",
            " #   Column   Non-Null Count   Dtype  \n",
            "---  ------   --------------   -----  \n",
            " 0   userId   100836 non-null  int64  \n",
            " 1   movieId  100836 non-null  int64  \n",
            " 2   rating   100836 non-null  float64\n",
            "dtypes: float64(1), int64(2)\n",
            "memory usage: 2.3 MB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ecnVM8lkih-W",
        "outputId": "edbdbd3e-3b97-42d0-82fd-facf97706814"
      },
      "source": [
        "# calculate the size of ratings matrix\n",
        "mov_cnt = rating_pd['movieId'].drop_duplicates().count()\n",
        "user_cnt = rating_pd['userId'].drop_duplicates().count()\n",
        "rate_cnt = rating_pd['rating'].count()\n",
        "matrix_size = user_cnt * mov_cnt\n",
        "percentage = rate_cnt/matrix_size * 100\n",
        "print(f'''\n",
        "From the ratings file, we obtained the info of ratings matrix as below:\n",
        "total movies are: {mov_cnt}\n",
        "total users are: {user_cnt}\n",
        "total ratings are: {rate_cnt}\n",
        "maxtrix size is {matrix_size}\n",
        "{percentage.round()}% of the matrix is filled.\n",
        "''')"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\n",
            "From the ratings file, we obtained the info of ratings matrix as below:\n",
            "total movies are: 9724\n",
            "total users are: 610\n",
            "total ratings are: 100836\n",
            "maxtrix size is 5931640\n",
            "2.0% of the matrix is filled.\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "G1AcrUuailSV"
      },
      "source": [
        "movie_df.registerTempTable(\"movie\")\n",
        "rating_df.registerTempTable(\"rating\")\n",
        "link_df.registerTempTable(\"link\")\n",
        "tag_df.registerTempTable(\"tag\")"
      ],
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qt911b-tipg0"
      },
      "source": [
        "### **1. OLAP** ### "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "E-Y_NGpTjEAS"
      },
      "source": [
        "#### **1.1 The Numbers of Users and Movies** ####"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 81
        },
        "id": "rbJsPJRpioWI",
        "outputId": "4d659d59-53ac-4e74-9c0f-26d2e66bfd49"
      },
      "source": [
        "user_cnt = spark.sql('select count(distinct userId) as Number_of_Users ' +\\\n",
        "                     'from rating').toPandas()\n",
        "user_cnt"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Number_of_Users</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>610</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Number_of_Users\n",
              "0              610"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 81
        },
        "id": "RAQs7gQIjQs8",
        "outputId": "0c487897-4d21-4d77-a83c-cac0fac62fcb"
      },
      "source": [
        "mov_cnt = spark.sql('select count(distinct movieId) as Movie_Count ' +\\\n",
        "                    'from movie').toPandas()\n",
        "mov_cnt"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Movie_Count</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9742</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Movie_Count\n",
              "0         9742"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7bJBLtVijXws"
      },
      "source": [
        "#### **1.2 How many movies are rated by users? List movies not rated before** ####"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 81
        },
        "id": "2Rc7hmdIjUQ6",
        "outputId": "c6dabf36-6ffa-41f9-9049-6369b1e296c8"
      },
      "source": [
        "rated_cnt = spark.sql('select count(distinct movieId) as Num_of_Rated_Movie ' +\\\n",
        "                      'from movie ' +\\\n",
        "                      'where movieId in (select movieId from rating)').toPandas()\n",
        "rated_cnt"
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Num_of_Rated_Movie</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9724</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Num_of_Rated_Movie\n",
              "0                9724"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 612
        },
        "id": "baWJrH_3jenT",
        "outputId": "5749bc3f-a186-4eae-d69e-c34b195ba20a"
      },
      "source": [
        "unrated_mov = spark.sql('select * ' +\\\n",
        "                        'from movie ' +\\\n",
        "                        'where movieId not in (select movieId from rating)').toPandas()\n",
        "unrated_mov"
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>movieId</th>\n",
              "      <th>title</th>\n",
              "      <th>genres</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1076</td>\n",
              "      <td>Innocents, The (1961)</td>\n",
              "      <td>Drama|Horror|Thriller</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2939</td>\n",
              "      <td>Niagara (1953)</td>\n",
              "      <td>Drama|Thriller</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3338</td>\n",
              "      <td>For All Mankind (1989)</td>\n",
              "      <td>Documentary</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3456</td>\n",
              "      <td>Color of Paradise, The (Rang-e khoda) (1999)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4194</td>\n",
              "      <td>I Know Where I'm Going! (1945)</td>\n",
              "      <td>Drama|Romance|War</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>5721</td>\n",
              "      <td>Chosen, The (1981)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>6668</td>\n",
              "      <td>Road Home, The (Wo de fu qin mu qin) (1999)</td>\n",
              "      <td>Drama|Romance</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>6849</td>\n",
              "      <td>Scrooge (1970)</td>\n",
              "      <td>Drama|Fantasy|Musical</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>7020</td>\n",
              "      <td>Proof (1991)</td>\n",
              "      <td>Comedy|Drama|Romance</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>7792</td>\n",
              "      <td>Parallax View, The (1974)</td>\n",
              "      <td>Thriller</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>8765</td>\n",
              "      <td>This Gun for Hire (1942)</td>\n",
              "      <td>Crime|Film-Noir|Thriller</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>11</th>\n",
              "      <td>25855</td>\n",
              "      <td>Roaring Twenties, The (1939)</td>\n",
              "      <td>Crime|Drama|Thriller</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12</th>\n",
              "      <td>26085</td>\n",
              "      <td>Mutiny on the Bounty (1962)</td>\n",
              "      <td>Adventure|Drama|Romance</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>13</th>\n",
              "      <td>30892</td>\n",
              "      <td>In the Realms of the Unreal (2004)</td>\n",
              "      <td>Animation|Documentary</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>14</th>\n",
              "      <td>32160</td>\n",
              "      <td>Twentieth Century (1934)</td>\n",
              "      <td>Comedy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>15</th>\n",
              "      <td>32371</td>\n",
              "      <td>Call Northside 777 (1948)</td>\n",
              "      <td>Crime|Drama|Film-Noir</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>16</th>\n",
              "      <td>34482</td>\n",
              "      <td>Browning Version, The (1951)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>17</th>\n",
              "      <td>85565</td>\n",
              "      <td>Chalet Girl (2011)</td>\n",
              "      <td>Comedy|Romance</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   movieId  ...                    genres\n",
              "0     1076  ...     Drama|Horror|Thriller\n",
              "1     2939  ...            Drama|Thriller\n",
              "2     3338  ...               Documentary\n",
              "3     3456  ...                     Drama\n",
              "4     4194  ...         Drama|Romance|War\n",
              "5     5721  ...                     Drama\n",
              "6     6668  ...             Drama|Romance\n",
              "7     6849  ...     Drama|Fantasy|Musical\n",
              "8     7020  ...      Comedy|Drama|Romance\n",
              "9     7792  ...                  Thriller\n",
              "10    8765  ...  Crime|Film-Noir|Thriller\n",
              "11   25855  ...      Crime|Drama|Thriller\n",
              "12   26085  ...   Adventure|Drama|Romance\n",
              "13   30892  ...     Animation|Documentary\n",
              "14   32160  ...                    Comedy\n",
              "15   32371  ...     Crime|Drama|Film-Noir\n",
              "16   34482  ...                     Drama\n",
              "17   85565  ...            Comedy|Romance\n",
              "\n",
              "[18 rows x 3 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HIyPhbpXjnub"
      },
      "source": [
        "#### **1.3 List Movie Genres** ####"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 675
        },
        "id": "_Ww24K_fjhf_",
        "outputId": "cc383bca-3649-4508-9c84-d6ef7bc33703"
      },
      "source": [
        "genre_df = spark.sql(\"select distinct explode(split(genres, '[|]')) as Genres \" +\\\n",
        "                     \"from movie \" +\\\n",
        "                     \"order by 1\").toPandas()\n",
        "genre_df"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Genres</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>(no genres listed)</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Action</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Adventure</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Animation</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Children</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>Comedy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>Crime</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>Documentary</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>Fantasy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>Film-Noir</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>11</th>\n",
              "      <td>Horror</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12</th>\n",
              "      <td>IMAX</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>13</th>\n",
              "      <td>Musical</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>14</th>\n",
              "      <td>Mystery</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>15</th>\n",
              "      <td>Romance</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>16</th>\n",
              "      <td>Sci-Fi</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>17</th>\n",
              "      <td>Thriller</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>18</th>\n",
              "      <td>War</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19</th>\n",
              "      <td>Western</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                Genres\n",
              "0   (no genres listed)\n",
              "1               Action\n",
              "2            Adventure\n",
              "3            Animation\n",
              "4             Children\n",
              "5               Comedy\n",
              "6                Crime\n",
              "7          Documentary\n",
              "8                Drama\n",
              "9              Fantasy\n",
              "10           Film-Noir\n",
              "11              Horror\n",
              "12                IMAX\n",
              "13             Musical\n",
              "14             Mystery\n",
              "15             Romance\n",
              "16              Sci-Fi\n",
              "17            Thriller\n",
              "18                 War\n",
              "19             Western"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mOUMst-QjwcT"
      },
      "source": [
        "#### **1.4 Movie for Each Category** ####"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 675
        },
        "id": "HqOq9KUOjsyl",
        "outputId": "87493233-431b-4a4a-f855-3b07fc9b935f"
      },
      "source": [
        "mov_per_cat = spark.sql(\"select Genres, count(movieId) as Num_of_Movies \" +\\\n",
        "                        \"from (select distinct explode(split(genres, '[|]')) as Genres, movieId from movie) \" +\\\n",
        "                        \"group by 1 \" +\\\n",
        "                        \"order by 2 desc\").toPandas()\n",
        "mov_per_cat"
      ],
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Genres</th>\n",
              "      <th>Num_of_Movies</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Drama</td>\n",
              "      <td>4361</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Comedy</td>\n",
              "      <td>3756</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Thriller</td>\n",
              "      <td>1894</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Action</td>\n",
              "      <td>1828</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Romance</td>\n",
              "      <td>1596</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>Adventure</td>\n",
              "      <td>1263</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>Crime</td>\n",
              "      <td>1199</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>Sci-Fi</td>\n",
              "      <td>980</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>Horror</td>\n",
              "      <td>978</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>Fantasy</td>\n",
              "      <td>779</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>Children</td>\n",
              "      <td>664</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>11</th>\n",
              "      <td>Animation</td>\n",
              "      <td>611</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12</th>\n",
              "      <td>Mystery</td>\n",
              "      <td>573</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>13</th>\n",
              "      <td>Documentary</td>\n",
              "      <td>440</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>14</th>\n",
              "      <td>War</td>\n",
              "      <td>382</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>15</th>\n",
              "      <td>Musical</td>\n",
              "      <td>334</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>16</th>\n",
              "      <td>Western</td>\n",
              "      <td>167</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>17</th>\n",
              "      <td>IMAX</td>\n",
              "      <td>158</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>18</th>\n",
              "      <td>Film-Noir</td>\n",
              "      <td>87</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19</th>\n",
              "      <td>(no genres listed)</td>\n",
              "      <td>34</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                Genres  Num_of_Movies\n",
              "0                Drama           4361\n",
              "1               Comedy           3756\n",
              "2             Thriller           1894\n",
              "3               Action           1828\n",
              "4              Romance           1596\n",
              "5            Adventure           1263\n",
              "6                Crime           1199\n",
              "7               Sci-Fi            980\n",
              "8               Horror            978\n",
              "9              Fantasy            779\n",
              "10            Children            664\n",
              "11           Animation            611\n",
              "12             Mystery            573\n",
              "13         Documentary            440\n",
              "14                 War            382\n",
              "15             Musical            334\n",
              "16             Western            167\n",
              "17                IMAX            158\n",
              "18           Film-Noir             87\n",
              "19  (no genres listed)             34"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 675
        },
        "id": "ymFJWNuqjzph",
        "outputId": "90c634f9-9cd6-45e5-914c-6f09c63b3759"
      },
      "source": [
        "mov_lst = spark.sql(\"select Genres, concat_ws(',', collect_set(title)) as Movie_List \" +\\\n",
        "                    \"from (select distinct explode(split(genres, '[|]')) as Genres, title from movie)\" +\\\n",
        "                    \"group by 1\").toPandas()\n",
        "mov_lst"
      ],
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Genres</th>\n",
              "      <th>Movie_List</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Crime</td>\n",
              "      <td>Stealing Rembrandt (Rembrandt) (2003),The Gamb...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Romance</td>\n",
              "      <td>Vampire in Brooklyn (1995),Hysteria (2011),Far...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Thriller</td>\n",
              "      <td>Element of Crime, The (Forbrydelsens Element) ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Adventure</td>\n",
              "      <td>Ice Age: Collision Course (2016),Masters of th...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Drama</td>\n",
              "      <td>Airport '77 (1977),Element of Crime, The (Forb...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>War</td>\n",
              "      <td>General, The (1926),Joyeux Noël (Merry Christm...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>Documentary</td>\n",
              "      <td>Jim &amp; Andy: The Great Beyond (2017),U2: Rattle...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>Fantasy</td>\n",
              "      <td>Masters of the Universe (1987),Odd Life of Tim...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>Mystery</td>\n",
              "      <td>Before and After (1996),Primal Fear (1996),'Sa...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>Musical</td>\n",
              "      <td>U2: Rattle and Hum (1988),Sword in the Stone, ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>Animation</td>\n",
              "      <td>Ice Age: Collision Course (2016),Planes (2013)...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>11</th>\n",
              "      <td>Film-Noir</td>\n",
              "      <td>Rififi (Du rififi chez les hommes) (1955),Swee...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12</th>\n",
              "      <td>(no genres listed)</td>\n",
              "      <td>T2 3-D: Battle Across Time (1996),A Midsummer ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>13</th>\n",
              "      <td>IMAX</td>\n",
              "      <td>Harry Potter and the Prisoner of Azkaban (2004...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>14</th>\n",
              "      <td>Horror</td>\n",
              "      <td>Tormented (1960),Vampire in Brooklyn (1995),'S...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>15</th>\n",
              "      <td>Western</td>\n",
              "      <td>Man Who Shot Liberty Valance, The (1962),Lone ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>16</th>\n",
              "      <td>Comedy</td>\n",
              "      <td>Hysteria (2011),Humpday (2009),Meet John Doe (...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>17</th>\n",
              "      <td>Children</td>\n",
              "      <td>Ice Age: Collision Course (2016),Nut Job, The ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>18</th>\n",
              "      <td>Action</td>\n",
              "      <td>Stealing Rembrandt (Rembrandt) (2003),Masters ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19</th>\n",
              "      <td>Sci-Fi</td>\n",
              "      <td>Push (2009),SORI: Voice from the Heart (2016),...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                Genres                                         Movie_List\n",
              "0                Crime  Stealing Rembrandt (Rembrandt) (2003),The Gamb...\n",
              "1              Romance  Vampire in Brooklyn (1995),Hysteria (2011),Far...\n",
              "2             Thriller  Element of Crime, The (Forbrydelsens Element) ...\n",
              "3            Adventure  Ice Age: Collision Course (2016),Masters of th...\n",
              "4                Drama  Airport '77 (1977),Element of Crime, The (Forb...\n",
              "5                  War  General, The (1926),Joyeux Noël (Merry Christm...\n",
              "6          Documentary  Jim & Andy: The Great Beyond (2017),U2: Rattle...\n",
              "7              Fantasy  Masters of the Universe (1987),Odd Life of Tim...\n",
              "8              Mystery  Before and After (1996),Primal Fear (1996),'Sa...\n",
              "9              Musical  U2: Rattle and Hum (1988),Sword in the Stone, ...\n",
              "10           Animation  Ice Age: Collision Course (2016),Planes (2013)...\n",
              "11           Film-Noir  Rififi (Du rififi chez les hommes) (1955),Swee...\n",
              "12  (no genres listed)  T2 3-D: Battle Across Time (1996),A Midsummer ...\n",
              "13                IMAX  Harry Potter and the Prisoner of Azkaban (2004...\n",
              "14              Horror  Tormented (1960),Vampire in Brooklyn (1995),'S...\n",
              "15             Western  Man Who Shot Liberty Valance, The (1962),Lone ...\n",
              "16              Comedy  Hysteria (2011),Humpday (2009),Meet John Doe (...\n",
              "17            Children  Ice Age: Collision Course (2016),Nut Job, The ...\n",
              "18              Action  Stealing Rembrandt (Rembrandt) (2003),Masters ...\n",
              "19              Sci-Fi  Push (2009),SORI: Voice from the Heart (2016),..."
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yG4PORbuj_OH"
      },
      "source": [
        "#### **1.5 Which rate did the majority users prefer? (optional)** ####"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 483
        },
        "id": "DjhzXTh1j6j3",
        "outputId": "b09f3874-8bfc-4364-f76b-02e998ac584e"
      },
      "source": [
        "plt.figure(figsize=(12, 8))\n",
        "sns.set_style('whitegrid')\n",
        "rating_pd_graph = rating_df.toPandas()\n",
        "plt.hist(rating_pd_graph.rating, color='purple')\n",
        "display()"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 864x576 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wqM76whmkHLY"
      },
      "source": [
        "#### **1.6 What are the top 20 movies that are rated most? (optional)** ####"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_MdNIUTLkDOO",
        "outputId": "cb2c1c33-7c50-496a-bf25-6cb4ae02f914"
      },
      "source": [
        "most_rated = spark.sql('select a.movieId, a.title, count(rating) as rating_cnt from movie a join rating b on a.movieId = b.movieId group by 1, 2 order by 3 desc limit 20')\n",
        "most_rated.show()"
      ],
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+-------+--------------------+----------+\n",
            "|movieId|               title|rating_cnt|\n",
            "+-------+--------------------+----------+\n",
            "|    356| Forrest Gump (1994)|       329|\n",
            "|    318|Shawshank Redempt...|       317|\n",
            "|    296| Pulp Fiction (1994)|       307|\n",
            "|    593|Silence of the La...|       279|\n",
            "|   2571|  Matrix, The (1999)|       278|\n",
            "|    260|Star Wars: Episod...|       251|\n",
            "|    480|Jurassic Park (1993)|       238|\n",
            "|    110|   Braveheart (1995)|       237|\n",
            "|    589|Terminator 2: Jud...|       224|\n",
            "|    527|Schindler's List ...|       220|\n",
            "|   2959|   Fight Club (1999)|       218|\n",
            "|      1|    Toy Story (1995)|       215|\n",
            "|   1196|Star Wars: Episod...|       211|\n",
            "|   2858|American Beauty (...|       204|\n",
            "|     50|Usual Suspects, T...|       204|\n",
            "|     47|Seven (a.k.a. Se7...|       203|\n",
            "|    780|Independence Day ...|       202|\n",
            "|    150|    Apollo 13 (1995)|       201|\n",
            "|   1198|Raiders of the Lo...|       200|\n",
            "|   4993|Lord of the Rings...|       198|\n",
            "+-------+--------------------+----------+\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZVF8cU65kUgO"
      },
      "source": [
        "### **2. Spark ALS based approach for training model** ### \n",
        "We will use an Spark ML to predict the ratings, so let's reload \"ratings.csv\" using sc.textFile and then convert it to the form of (user, item, rating) tuples."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p1JZPc_ukLzo",
        "outputId": "01a84c70-e0e5-4265-f30a-54c7fc852e4e"
      },
      "source": [
        "rating_df.printSchema()"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "root\n",
            " |-- userId: string (nullable = true)\n",
            " |-- movieId: string (nullable = true)\n",
            " |-- rating: string (nullable = true)\n",
            " |-- timestamp: string (nullable = true)\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hcUwQSWykj9-",
        "outputId": "2978acae-c400-4bbd-c7ed-030015719ef9"
      },
      "source": [
        "#drop unnecessary column \n",
        "rating_ndf =rating_df.drop('timestamp')\n",
        "\n",
        "#convert types to desired ones by pyspark dataframe formats\n",
        "from pyspark.sql.types import IntegerType, FloatType\n",
        "rating_ndf = rating_ndf.withColumn(\"userId\", rating_ndf[\"userId\"].cast(IntegerType()))\n",
        "rating_ndf = rating_ndf.withColumn(\"movieId\", rating_ndf[\"movieId\"].cast(IntegerType()))\n",
        "rating_ndf = rating_ndf.withColumn(\"rating\", rating_ndf[\"rating\"].cast(FloatType()))\n",
        "rating_ndf.show()\n",
        "rating_ndf.printSchema()"
      ],
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "+------+-------+------+\n",
            "|userId|movieId|rating|\n",
            "+------+-------+------+\n",
            "|     1|      1|   4.0|\n",
            "|     1|      3|   4.0|\n",
            "|     1|      6|   4.0|\n",
            "|     1|     47|   5.0|\n",
            "|     1|     50|   5.0|\n",
            "|     1|     70|   3.0|\n",
            "|     1|    101|   5.0|\n",
            "|     1|    110|   4.0|\n",
            "|     1|    151|   5.0|\n",
            "|     1|    157|   5.0|\n",
            "|     1|    163|   5.0|\n",
            "|     1|    216|   5.0|\n",
            "|     1|    223|   3.0|\n",
            "|     1|    231|   5.0|\n",
            "|     1|    235|   4.0|\n",
            "|     1|    260|   5.0|\n",
            "|     1|    296|   3.0|\n",
            "|     1|    316|   3.0|\n",
            "|     1|    333|   5.0|\n",
            "|     1|    349|   4.0|\n",
            "+------+-------+------+\n",
            "only showing top 20 rows\n",
            "\n",
            "root\n",
            " |-- userId: integer (nullable = true)\n",
            " |-- movieId: integer (nullable = true)\n",
            " |-- rating: float (nullable = true)\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nucutqM4kq3t"
      },
      "source": [
        "#### **2.1 ALS Model Selection and Evaluation** ####\n",
        "With the ALS model, we can use a grid search to find the optimal hyperparameters."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-cm4NY2Jkm7k"
      },
      "source": [
        "# import package\n",
        "from pyspark.ml.evaluation import RegressionEvaluator\n",
        "from pyspark.ml.recommendation import ALS\n",
        "from pyspark.ml.tuning import CrossValidator, ParamGridBuilder"
      ],
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "sMOKcu2wlOR7"
      },
      "source": [
        "# create test and train set\n",
        "(training, test) = rating_ndf.randomSplit([0.8,0.2])"
      ],
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6BdmN4yulRpi"
      },
      "source": [
        "# create ALS model\n",
        "# set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics\n",
        "from pyspark.ml.recommendation import ALS\n",
        "als = ALS(maxIter=10, rank=10, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy=\"drop\")"
      ],
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "B0ChWNrzlUVh"
      },
      "source": [
        "# tune model using ParamGridBuilder\n",
        "paramGrid = (ParamGridBuilder()\n",
        "             .addGrid(als.regParam, [0.05, 0.1, 0.3, 0.5])\n",
        "             .addGrid(als.rank, [5, 10, 15])\n",
        "             .addGrid(als.maxIter, [1, 5, 10])\n",
        "             .build())"
      ],
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8LJGUS74lWkf"
      },
      "source": [
        "# define evaluator as RMSE\n",
        "evaluator = RegressionEvaluator(metricName=\"rmse\", labelCol=\"rating\", predictionCol=\"prediction\")"
      ],
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mTfe6hHclYw_"
      },
      "source": [
        "# build Cross validation \n",
        "cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)"
      ],
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tZmv5XC7lbKj"
      },
      "source": [
        "# fit ALS model to training data\n",
        "cvModel = cv.fit(training)"
      ],
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WhCms4B2lgqN"
      },
      "source": [
        "# extract best model from the tuning exercise using ParamGridBuilder\n",
        "bestModel=cvModel.bestModel"
      ],
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VHn3O2PMmJHa"
      },
      "source": [
        "#### **2.2 Model Testing** ####\n",
        "Finally, make a prediction and check the testing error."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oquEDUW7loLh"
      },
      "source": [
        "# generate predictions and evaluate using RMSE\n",
        "predictions=bestModel.transform(test)\n",
        "rmse = evaluator.evaluate(predictions)"
      ],
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2mjaOzoju8R8",
        "outputId": "333b1499-fccd-4b10-f1b1-b21b16af9b62"
      },
      "source": [
        "# print evaluation metrics and model parameters\n",
        "print(\"RMSE =\" + str(rmse))\n",
        "print(\" ** Best Model** \")\n",
        "print(\" Rank = \" + str(bestModel._java_obj.parent().getRank()))\n",
        "print(\" MaxIter =\" + str(bestModel._java_obj.parent().getMaxIter()))\n",
        "print(\" RegParam = \" + str(bestModel._java_obj.parent().getRegParam()))"
      ],
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "RMSE =0.8812061645142741\n",
            " ** Best Model** \n",
            " Rank = 5\n",
            " MaxIter =10\n",
            " RegParam = 0.1\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 422
        },
        "id": "t0YUixgUvBQm",
        "outputId": "46eb5964-8a2e-4312-b310-a9dd812b0e46"
      },
      "source": [
        "predictions.toPandas()"
      ],
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>userId</th>\n",
              "      <th>movieId</th>\n",
              "      <th>rating</th>\n",
              "      <th>prediction</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>409</td>\n",
              "      <td>471</td>\n",
              "      <td>3.0</td>\n",
              "      <td>4.198646</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>372</td>\n",
              "      <td>471</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.443952</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>599</td>\n",
              "      <td>471</td>\n",
              "      <td>2.5</td>\n",
              "      <td>2.531919</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>603</td>\n",
              "      <td>471</td>\n",
              "      <td>4.0</td>\n",
              "      <td>3.164718</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>218</td>\n",
              "      <td>471</td>\n",
              "      <td>4.0</td>\n",
              "      <td>2.371195</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19523</th>\n",
              "      <td>249</td>\n",
              "      <td>79008</td>\n",
              "      <td>4.5</td>\n",
              "      <td>4.036290</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19524</th>\n",
              "      <td>563</td>\n",
              "      <td>84374</td>\n",
              "      <td>2.5</td>\n",
              "      <td>3.332607</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19525</th>\n",
              "      <td>298</td>\n",
              "      <td>84374</td>\n",
              "      <td>0.5</td>\n",
              "      <td>1.600757</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19526</th>\n",
              "      <td>562</td>\n",
              "      <td>84374</td>\n",
              "      <td>3.5</td>\n",
              "      <td>3.266908</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19527</th>\n",
              "      <td>525</td>\n",
              "      <td>147378</td>\n",
              "      <td>3.5</td>\n",
              "      <td>2.670312</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>19528 rows × 4 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "       userId  movieId  rating  prediction\n",
              "0         409      471     3.0    4.198646\n",
              "1         372      471     3.0    3.443952\n",
              "2         599      471     2.5    2.531919\n",
              "3         603      471     4.0    3.164718\n",
              "4         218      471     4.0    2.371195\n",
              "...       ...      ...     ...         ...\n",
              "19523     249    79008     4.5    4.036290\n",
              "19524     563    84374     2.5    3.332607\n",
              "19525     298    84374     0.5    1.600757\n",
              "19526     562    84374     3.5    3.266908\n",
              "19527     525   147378     3.5    2.670312\n",
              "\n",
              "[19528 rows x 4 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0FzdoBfnvTPm"
      },
      "source": [
        "#### **2.3 Model Apply and Check the Performance** ####"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0HWA1oBMvH77",
        "outputId": "c044c5ec-2451-4647-b265-7a75686cce60"
      },
      "source": [
        "alldata = bestModel.transform(rating_ndf)\n",
        "rmse = evaluator.evaluate(alldata)\n",
        "print(\"RMSE (for whole dataset) = \" + str(rmse))"
      ],
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "RMSE (for whole dataset) = 0.6927761542800891\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 422
        },
        "id": "LSZQ-GNdvytQ",
        "outputId": "2160d146-716f-4080-f663-956d7cbe96b7"
      },
      "source": [
        "alldata.registerTempTable('alldata_view')\n",
        "alldata.toPandas()"
      ],
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>userId</th>\n",
              "      <th>movieId</th>\n",
              "      <th>rating</th>\n",
              "      <th>prediction</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>191</td>\n",
              "      <td>148</td>\n",
              "      <td>5.0</td>\n",
              "      <td>4.933486</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>133</td>\n",
              "      <td>471</td>\n",
              "      <td>4.0</td>\n",
              "      <td>3.162882</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>597</td>\n",
              "      <td>471</td>\n",
              "      <td>2.0</td>\n",
              "      <td>4.087759</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>385</td>\n",
              "      <td>471</td>\n",
              "      <td>4.0</td>\n",
              "      <td>3.312406</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>436</td>\n",
              "      <td>471</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.612757</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>100057</th>\n",
              "      <td>567</td>\n",
              "      <td>145839</td>\n",
              "      <td>1.5</td>\n",
              "      <td>1.477010</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>100058</th>\n",
              "      <td>210</td>\n",
              "      <td>147378</td>\n",
              "      <td>4.5</td>\n",
              "      <td>4.061054</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>100059</th>\n",
              "      <td>380</td>\n",
              "      <td>147378</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.143582</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>100060</th>\n",
              "      <td>525</td>\n",
              "      <td>147378</td>\n",
              "      <td>3.5</td>\n",
              "      <td>2.670312</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>100061</th>\n",
              "      <td>517</td>\n",
              "      <td>147378</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.135213</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>100062 rows × 4 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "        userId  movieId  rating  prediction\n",
              "0          191      148     5.0    4.933486\n",
              "1          133      471     4.0    3.162882\n",
              "2          597      471     2.0    4.087759\n",
              "3          385      471     4.0    3.312406\n",
              "4          436      471     3.0    3.612757\n",
              "...        ...      ...     ...         ...\n",
              "100057     567   145839     1.5    1.477010\n",
              "100058     210   147378     4.5    4.061054\n",
              "100059     380   147378     3.0    3.143582\n",
              "100060     525   147378     3.5    2.670312\n",
              "100061     517   147378     1.0    1.135213\n",
              "\n",
              "[100062 rows x 4 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 491
        },
        "id": "hDpuvS9n0Kfm",
        "outputId": "5a5ef355-e5a0-415f-d61a-0c928752bc90"
      },
      "source": [
        "mov_rating_df = spark.sql('select b.*, a.title, a.genres ' +\\\n",
        "                          'from movie a join alldata_view b on a.movieId=b.movieId')\n",
        "mov_rating_df.toPandas()"
      ],
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>userId</th>\n",
              "      <th>movieId</th>\n",
              "      <th>rating</th>\n",
              "      <th>prediction</th>\n",
              "      <th>title</th>\n",
              "      <th>genres</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>191</td>\n",
              "      <td>148</td>\n",
              "      <td>5.0</td>\n",
              "      <td>4.933486</td>\n",
              "      <td>Awfully Big Adventure, An (1995)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>133</td>\n",
              "      <td>471</td>\n",
              "      <td>4.0</td>\n",
              "      <td>3.162882</td>\n",
              "      <td>Hudsucker Proxy, The (1994)</td>\n",
              "      <td>Comedy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>597</td>\n",
              "      <td>471</td>\n",
              "      <td>2.0</td>\n",
              "      <td>4.087759</td>\n",
              "      <td>Hudsucker Proxy, The (1994)</td>\n",
              "      <td>Comedy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>385</td>\n",
              "      <td>471</td>\n",
              "      <td>4.0</td>\n",
              "      <td>3.312406</td>\n",
              "      <td>Hudsucker Proxy, The (1994)</td>\n",
              "      <td>Comedy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>436</td>\n",
              "      <td>471</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.612757</td>\n",
              "      <td>Hudsucker Proxy, The (1994)</td>\n",
              "      <td>Comedy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>100057</th>\n",
              "      <td>567</td>\n",
              "      <td>145839</td>\n",
              "      <td>1.5</td>\n",
              "      <td>1.477010</td>\n",
              "      <td>Concussion (2015)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>100058</th>\n",
              "      <td>210</td>\n",
              "      <td>147378</td>\n",
              "      <td>4.5</td>\n",
              "      <td>4.061054</td>\n",
              "      <td>Doctor Who: Planet of the Dead (2009)</td>\n",
              "      <td>Adventure|Children|Drama|Sci-Fi</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>100059</th>\n",
              "      <td>380</td>\n",
              "      <td>147378</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.143582</td>\n",
              "      <td>Doctor Who: Planet of the Dead (2009)</td>\n",
              "      <td>Adventure|Children|Drama|Sci-Fi</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>100060</th>\n",
              "      <td>525</td>\n",
              "      <td>147378</td>\n",
              "      <td>3.5</td>\n",
              "      <td>2.670312</td>\n",
              "      <td>Doctor Who: Planet of the Dead (2009)</td>\n",
              "      <td>Adventure|Children|Drama|Sci-Fi</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>100061</th>\n",
              "      <td>517</td>\n",
              "      <td>147378</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.135213</td>\n",
              "      <td>Doctor Who: Planet of the Dead (2009)</td>\n",
              "      <td>Adventure|Children|Drama|Sci-Fi</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>100062 rows × 6 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "        userId  ...                           genres\n",
              "0          191  ...                            Drama\n",
              "1          133  ...                           Comedy\n",
              "2          597  ...                           Comedy\n",
              "3          385  ...                           Comedy\n",
              "4          436  ...                           Comedy\n",
              "...        ...  ...                              ...\n",
              "100057     567  ...                            Drama\n",
              "100058     210  ...  Adventure|Children|Drama|Sci-Fi\n",
              "100059     380  ...  Adventure|Children|Drama|Sci-Fi\n",
              "100060     525  ...  Adventure|Children|Drama|Sci-Fi\n",
              "100061     517  ...  Adventure|Children|Drama|Sci-Fi\n",
              "\n",
              "[100062 rows x 6 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 42
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "01pBfujUzj0c"
      },
      "source": [
        "### **3. Make Recommendations** ###\n",
        "Find the similar moives based on the ALS results"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fzS2U_b20SNX"
      },
      "source": [
        "#### **3.1 Find the similar moives for users with id: 575， 232** ####"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-v5ldXJrQkl_"
      },
      "source": [
        "- Method 1: create a function to wrap up model, user and quantity of recommended movies\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6hpbocCF1UYs"
      },
      "source": [
        "from pyspark.sql.functions import lit\n",
        "#lit: Creates a Column of literal value\n",
        "\n",
        "def userRecs1(model, user, quantity):\n",
        "    # the specified user and all the movies listed in the ratings\n",
        "    userSet = mov_rating_df.select('movieId').distinct().withColumn('userId', lit(user))\n",
        "    \n",
        "    # the movies that have already been rated by this user\n",
        "    ratedSet = mov_rating_df.filter(mov_rating_df.userId == user).select('movieId', 'userId')\n",
        "    \n",
        "    # apply the recommender system to the dataset without the already-rated movies to predict ratings\n",
        "    predictions = model.transform(userSet.subtract(ratedSet)).dropna().orderBy('prediction', ascending=False).limit(quantity).select('movieId', 'prediction')\n",
        "    \n",
        "    # Join with the movies_df to get the movies titles and genres\n",
        "    recommendations = predictions.join(movie_df, predictions.movieId == movie_df.movieId).select(predictions.movieId, movie_df.title, movie_df.genres, predictions.prediction)\n",
        "    return recommendations"
      ],
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 362
        },
        "id": "jlpsEZ8T0s79",
        "outputId": "038f1787-505c-4b40-ef3b-94b78db89f20"
      },
      "source": [
        "userRecs1(bestModel, 575, 10).toPandas()"
      ],
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>movieId</th>\n",
              "      <th>title</th>\n",
              "      <th>genres</th>\n",
              "      <th>prediction</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>441</td>\n",
              "      <td>Dazed and Confused (1993)</td>\n",
              "      <td>Comedy</td>\n",
              "      <td>5.032527</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1237</td>\n",
              "      <td>Seventh Seal, The (Sjunde inseglet, Det) (1957)</td>\n",
              "      <td>Drama</td>\n",
              "      <td>5.052160</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>5075</td>\n",
              "      <td>Waydowntown (2000)</td>\n",
              "      <td>Comedy</td>\n",
              "      <td>5.179841</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>6086</td>\n",
              "      <td>I, the Jury (1982)</td>\n",
              "      <td>Crime|Drama|Thriller</td>\n",
              "      <td>5.087701</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>7096</td>\n",
              "      <td>Rivers and Tides (2001)</td>\n",
              "      <td>Documentary</td>\n",
              "      <td>5.098831</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>7841</td>\n",
              "      <td>Children of Dune (2003)</td>\n",
              "      <td>Fantasy|Sci-Fi</td>\n",
              "      <td>5.476450</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>33649</td>\n",
              "      <td>Saving Face (2004)</td>\n",
              "      <td>Comedy|Drama|Romance</td>\n",
              "      <td>5.124772</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>58303</td>\n",
              "      <td>Counterfeiters, The (Die Fälscher) (2007)</td>\n",
              "      <td>Crime|Drama|War</td>\n",
              "      <td>5.169242</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>59018</td>\n",
              "      <td>Visitor, The (2007)</td>\n",
              "      <td>Drama|Romance</td>\n",
              "      <td>5.411258</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>60943</td>\n",
              "      <td>Frozen River (2008)</td>\n",
              "      <td>Drama</td>\n",
              "      <td>5.411258</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   movieId  ... prediction\n",
              "0      441  ...   5.032527\n",
              "1     1237  ...   5.052160\n",
              "2     5075  ...   5.179841\n",
              "3     6086  ...   5.087701\n",
              "4     7096  ...   5.098831\n",
              "5     7841  ...   5.476450\n",
              "6    33649  ...   5.124772\n",
              "7    58303  ...   5.169242\n",
              "8    59018  ...   5.411258\n",
              "9    60943  ...   5.411258\n",
              "\n",
              "[10 rows x 4 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 44
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 362
        },
        "id": "cH-xrXxq8BTq",
        "outputId": "fe2f491d-d81f-4d74-de5a-6827b367368e"
      },
      "source": [
        "userRecs1(bestModel, 232, 10).toPandas()"
      ],
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>movieId</th>\n",
              "      <th>title</th>\n",
              "      <th>genres</th>\n",
              "      <th>prediction</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>7071</td>\n",
              "      <td>Woman Under the Influence, A (1974)</td>\n",
              "      <td>Drama</td>\n",
              "      <td>4.631033</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>26073</td>\n",
              "      <td>Human Condition III, The (Ningen no joken III)...</td>\n",
              "      <td>Drama|War</td>\n",
              "      <td>4.631033</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>26928</td>\n",
              "      <td>Summer's Tale, A (Conte d'été) (1996)</td>\n",
              "      <td>Comedy|Drama|Romance</td>\n",
              "      <td>4.631033</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>33649</td>\n",
              "      <td>Saving Face (2004)</td>\n",
              "      <td>Comedy|Drama|Romance</td>\n",
              "      <td>4.880827</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>74226</td>\n",
              "      <td>Dream of Light (a.k.a. Quince Tree Sun, The) (...</td>\n",
              "      <td>Documentary|Drama</td>\n",
              "      <td>4.631033</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>84273</td>\n",
              "      <td>Zeitgeist: Moving Forward (2011)</td>\n",
              "      <td>Documentary</td>\n",
              "      <td>4.631033</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>86237</td>\n",
              "      <td>Connections (1978)</td>\n",
              "      <td>Documentary</td>\n",
              "      <td>4.631033</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>93988</td>\n",
              "      <td>North &amp; South (2004)</td>\n",
              "      <td>Drama|Romance</td>\n",
              "      <td>4.656671</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>134796</td>\n",
              "      <td>Bitter Lake (2015)</td>\n",
              "      <td>Documentary</td>\n",
              "      <td>4.631033</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>138966</td>\n",
              "      <td>Nasu: Summer in Andalusia (2003)</td>\n",
              "      <td>Animation</td>\n",
              "      <td>4.631033</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   movieId  ... prediction\n",
              "0     7071  ...   4.631033\n",
              "1    26073  ...   4.631033\n",
              "2    26928  ...   4.631033\n",
              "3    33649  ...   4.880827\n",
              "4    74226  ...   4.631033\n",
              "5    84273  ...   4.631033\n",
              "6    86237  ...   4.631033\n",
              "7    93988  ...   4.656671\n",
              "8   134796  ...   4.631033\n",
              "9   138966  ...   4.631033\n",
              "\n",
              "[10 rows x 4 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 45
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fnoiY5TqQ_ms"
      },
      "source": [
        "- Method 2: recommend by ALS api \"recommendForAllUsers()\""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 422
        },
        "id": "C2JkRMpMQ_F-",
        "outputId": "d60429b9-1f89-450c-c9e1-eb2cdd1d86a2"
      },
      "source": [
        "userRecs2 = bestModel.recommendForAllUsers(10)\n",
        "userRecs2.toPandas()"
      ],
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>userId</th>\n",
              "      <th>recommendations</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>471</td>\n",
              "      <td>[(89904, 4.959934711456299), (8235, 4.86559104...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>463</td>\n",
              "      <td>[(6818, 5.416696071624756), (33649, 5.38040208...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>496</td>\n",
              "      <td>[(6818, 5.079161167144775), (3266, 4.965522766...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>148</td>\n",
              "      <td>[(49347, 4.969733715057373), (51931, 4.9583921...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>540</td>\n",
              "      <td>[(6818, 5.956073760986328), (25771, 5.59065532...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>605</th>\n",
              "      <td>208</td>\n",
              "      <td>[(6818, 4.963436603546143), (25771, 4.94961690...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>606</th>\n",
              "      <td>401</td>\n",
              "      <td>[(5666, 4.849633693695068), (1046, 4.500442028...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>607</th>\n",
              "      <td>422</td>\n",
              "      <td>[(6818, 4.9726176261901855), (33649, 4.7847967...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>608</th>\n",
              "      <td>517</td>\n",
              "      <td>[(32892, 4.854342460632324), (3283, 4.37140750...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>609</th>\n",
              "      <td>89</td>\n",
              "      <td>[(69211, 4.931743144989014), (121781, 4.931743...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>610 rows × 2 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "     userId                                    recommendations\n",
              "0       471  [(89904, 4.959934711456299), (8235, 4.86559104...\n",
              "1       463  [(6818, 5.416696071624756), (33649, 5.38040208...\n",
              "2       496  [(6818, 5.079161167144775), (3266, 4.965522766...\n",
              "3       148  [(49347, 4.969733715057373), (51931, 4.9583921...\n",
              "4       540  [(6818, 5.956073760986328), (25771, 5.59065532...\n",
              "..      ...                                                ...\n",
              "605     208  [(6818, 4.963436603546143), (25771, 4.94961690...\n",
              "606     401  [(5666, 4.849633693695068), (1046, 4.500442028...\n",
              "607     422  [(6818, 4.9726176261901855), (33649, 4.7847967...\n",
              "608     517  [(32892, 4.854342460632324), (3283, 4.37140750...\n",
              "609      89  [(69211, 4.931743144989014), (121781, 4.931743...\n",
              "\n",
              "[610 rows x 2 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 46
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 422
        },
        "id": "GUE6GFmbRQq3",
        "outputId": "84de4621-7bbb-4d08-9566-b59bacee3a96"
      },
      "source": [
        "from pyspark.sql.functions import explode, col\n",
        "userRecs2 = userRecs2\\\n",
        "    .withColumn(\"rec_exp\", explode(\"recommendations\"))\\\n",
        "    .select('userId', col(\"rec_exp.movieId\"), col(\"rec_exp.rating\"))\n",
        "userRecs2.toPandas()"
      ],
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>userId</th>\n",
              "      <th>movieId</th>\n",
              "      <th>rating</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>471</td>\n",
              "      <td>89904</td>\n",
              "      <td>4.959935</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>471</td>\n",
              "      <td>8235</td>\n",
              "      <td>4.865591</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>471</td>\n",
              "      <td>4495</td>\n",
              "      <td>4.865591</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>471</td>\n",
              "      <td>51931</td>\n",
              "      <td>4.761309</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>471</td>\n",
              "      <td>158966</td>\n",
              "      <td>4.743157</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6095</th>\n",
              "      <td>89</td>\n",
              "      <td>136341</td>\n",
              "      <td>4.931743</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6096</th>\n",
              "      <td>89</td>\n",
              "      <td>53280</td>\n",
              "      <td>4.931743</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6097</th>\n",
              "      <td>89</td>\n",
              "      <td>131130</td>\n",
              "      <td>4.931743</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6098</th>\n",
              "      <td>89</td>\n",
              "      <td>147410</td>\n",
              "      <td>4.931743</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6099</th>\n",
              "      <td>89</td>\n",
              "      <td>44851</td>\n",
              "      <td>4.931743</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>6100 rows × 3 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "      userId  movieId    rating\n",
              "0        471    89904  4.959935\n",
              "1        471     8235  4.865591\n",
              "2        471     4495  4.865591\n",
              "3        471    51931  4.761309\n",
              "4        471   158966  4.743157\n",
              "...      ...      ...       ...\n",
              "6095      89   136341  4.931743\n",
              "6096      89    53280  4.931743\n",
              "6097      89   131130  4.931743\n",
              "6098      89   147410  4.931743\n",
              "6099      89    44851  4.931743\n",
              "\n",
              "[6100 rows x 3 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 47
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 362
        },
        "id": "fF4-IIJSRTQ7",
        "outputId": "67cd9085-cd06-40c0-c436-c23c1f1ed415"
      },
      "source": [
        "userRecs2.join(movie_df, on='movieId').filter('userId = 575').toPandas()"
      ],
      "execution_count": 48,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>movieId</th>\n",
              "      <th>userId</th>\n",
              "      <th>rating</th>\n",
              "      <th>title</th>\n",
              "      <th>genres</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>7841</td>\n",
              "      <td>575</td>\n",
              "      <td>5.476450</td>\n",
              "      <td>Children of Dune (2003)</td>\n",
              "      <td>Fantasy|Sci-Fi</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>59018</td>\n",
              "      <td>575</td>\n",
              "      <td>5.411258</td>\n",
              "      <td>Visitor, The (2007)</td>\n",
              "      <td>Drama|Romance</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>60943</td>\n",
              "      <td>575</td>\n",
              "      <td>5.411258</td>\n",
              "      <td>Frozen River (2008)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5075</td>\n",
              "      <td>575</td>\n",
              "      <td>5.179841</td>\n",
              "      <td>Waydowntown (2000)</td>\n",
              "      <td>Comedy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>58303</td>\n",
              "      <td>575</td>\n",
              "      <td>5.169242</td>\n",
              "      <td>Counterfeiters, The (Die Fälscher) (2007)</td>\n",
              "      <td>Crime|Drama|War</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>33649</td>\n",
              "      <td>575</td>\n",
              "      <td>5.124772</td>\n",
              "      <td>Saving Face (2004)</td>\n",
              "      <td>Comedy|Drama|Romance</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>7096</td>\n",
              "      <td>575</td>\n",
              "      <td>5.098831</td>\n",
              "      <td>Rivers and Tides (2001)</td>\n",
              "      <td>Documentary</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>6086</td>\n",
              "      <td>575</td>\n",
              "      <td>5.087701</td>\n",
              "      <td>I, the Jury (1982)</td>\n",
              "      <td>Crime|Drama|Thriller</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>1237</td>\n",
              "      <td>575</td>\n",
              "      <td>5.052160</td>\n",
              "      <td>Seventh Seal, The (Sjunde inseglet, Det) (1957)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>441</td>\n",
              "      <td>575</td>\n",
              "      <td>5.032527</td>\n",
              "      <td>Dazed and Confused (1993)</td>\n",
              "      <td>Comedy</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   movieId  ...                genres\n",
              "0     7841  ...        Fantasy|Sci-Fi\n",
              "1    59018  ...         Drama|Romance\n",
              "2    60943  ...                 Drama\n",
              "3     5075  ...                Comedy\n",
              "4    58303  ...       Crime|Drama|War\n",
              "5    33649  ...  Comedy|Drama|Romance\n",
              "6     7096  ...           Documentary\n",
              "7     6086  ...  Crime|Drama|Thriller\n",
              "8     1237  ...                 Drama\n",
              "9      441  ...                Comedy\n",
              "\n",
              "[10 rows x 5 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 48
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 362
        },
        "id": "BC865vpeRvJW",
        "outputId": "a633f14d-aabe-4c89-d4fc-6d769210f092"
      },
      "source": [
        "userRecs2.join(movie_df, on='movieId').filter('userId = 232').toPandas()"
      ],
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>movieId</th>\n",
              "      <th>userId</th>\n",
              "      <th>rating</th>\n",
              "      <th>title</th>\n",
              "      <th>genres</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>33649</td>\n",
              "      <td>232</td>\n",
              "      <td>4.880827</td>\n",
              "      <td>Saving Face (2004)</td>\n",
              "      <td>Comedy|Drama|Romance</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>93988</td>\n",
              "      <td>232</td>\n",
              "      <td>4.656671</td>\n",
              "      <td>North &amp; South (2004)</td>\n",
              "      <td>Drama|Romance</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>5666</td>\n",
              "      <td>232</td>\n",
              "      <td>4.646063</td>\n",
              "      <td>Rules of Attraction, The (2002)</td>\n",
              "      <td>Comedy|Drama|Romance|Thriller</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>184245</td>\n",
              "      <td>232</td>\n",
              "      <td>4.631033</td>\n",
              "      <td>De platte jungle (1978)</td>\n",
              "      <td>Documentary</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>179135</td>\n",
              "      <td>232</td>\n",
              "      <td>4.631033</td>\n",
              "      <td>Blue Planet II (2017)</td>\n",
              "      <td>Documentary</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>138966</td>\n",
              "      <td>232</td>\n",
              "      <td>4.631033</td>\n",
              "      <td>Nasu: Summer in Andalusia (2003)</td>\n",
              "      <td>Animation</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>7071</td>\n",
              "      <td>232</td>\n",
              "      <td>4.631033</td>\n",
              "      <td>Woman Under the Influence, A (1974)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>74226</td>\n",
              "      <td>232</td>\n",
              "      <td>4.631033</td>\n",
              "      <td>Dream of Light (a.k.a. Quince Tree Sun, The) (...</td>\n",
              "      <td>Documentary|Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>84273</td>\n",
              "      <td>232</td>\n",
              "      <td>4.631033</td>\n",
              "      <td>Zeitgeist: Moving Forward (2011)</td>\n",
              "      <td>Documentary</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>26073</td>\n",
              "      <td>232</td>\n",
              "      <td>4.631033</td>\n",
              "      <td>Human Condition III, The (Ningen no joken III)...</td>\n",
              "      <td>Drama|War</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   movieId  ...                         genres\n",
              "0    33649  ...           Comedy|Drama|Romance\n",
              "1    93988  ...                  Drama|Romance\n",
              "2     5666  ...  Comedy|Drama|Romance|Thriller\n",
              "3   184245  ...                    Documentary\n",
              "4   179135  ...                    Documentary\n",
              "5   138966  ...                      Animation\n",
              "6     7071  ...                          Drama\n",
              "7    74226  ...              Documentary|Drama\n",
              "8    84273  ...                    Documentary\n",
              "9    26073  ...                      Drama|War\n",
              "\n",
              "[10 rows x 5 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 49
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZWrHAOHr_Yyt"
      },
      "source": [
        "#### **3.2 Find the similar moives for moive with id: 471, 40491** ####"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Sbug4QJW_MGH"
      },
      "source": [
        "from pyspark.ml.linalg import *\n",
        "from pyspark.sql.types import * "
      ],
      "execution_count": 50,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zD9nkWFf_lNw",
        "outputId": "aac82c0f-dc5a-4f45-f1c2-25a590b1695e"
      },
      "source": [
        "itemFactors = bestModel.itemFactors\n",
        "itemFactors.printSchema()\n",
        "itemFactors.show()"
      ],
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "root\n",
            " |-- id: integer (nullable = false)\n",
            " |-- features: array (nullable = true)\n",
            " |    |-- element: float (containsNull = false)\n",
            "\n",
            "+---+--------------------+\n",
            "| id|            features|\n",
            "+---+--------------------+\n",
            "| 10|[1.2477971, -1.23...|\n",
            "| 20|[1.2827852, -0.96...|\n",
            "| 30|[0.6489236, -0.93...|\n",
            "| 40|[0.9574917, -1.02...|\n",
            "| 50|[1.6372948, -1.51...|\n",
            "| 60|[1.2362996, -1.29...|\n",
            "| 70|[1.208497, -1.227...|\n",
            "| 80|[1.4761215, -1.12...|\n",
            "|100|[1.4798532, -0.68...|\n",
            "|110|[1.4177991, -1.76...|\n",
            "|140|[1.6373909, -0.87...|\n",
            "|150|[1.4349048, -1.56...|\n",
            "|160|[1.2032279, -0.97...|\n",
            "|170|[1.4931022, -1.06...|\n",
            "|180|[0.9167522, -1.45...|\n",
            "|190|[1.6928645, -0.66...|\n",
            "|210|[1.282163, -1.115...|\n",
            "|220|[0.91061413, -1.2...|\n",
            "|230|[1.3890396, -1.43...|\n",
            "|240|[0.78188866, -0.9...|\n",
            "+---+--------------------+\n",
            "only showing top 20 rows\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "I3OsmR6x_pK1"
      },
      "source": [
        "def similarMovies(inputId):\n",
        "  #built a DataFrame for similar movie\n",
        "  similarMovie=pd.DataFrame(columns=('movieId','cosine_similarity'))\n",
        "  \n",
        "  # retrieves all elements in a DataFrame(itemFactors) as an array to the driver.\n",
        "  movieFeat = itemFactors.filter(itemFactors.id == inputId).select('features').collect()\n",
        "  \n",
        "  for id, features in itemFactors.collect():\n",
        "    cs = np.dot(movieFeat, features) / (np.linalg.norm(movieFeat) * np.linalg.norm(features))\n",
        "    similarMovie=similarMovie.append({'movieId':str(id), 'cosine_similarity':cs}, ignore_index=True)\n",
        "    similarMovie_cs = similarMovie.sort_values(by=['cosine_similarity'],ascending = False)[1:11]\n",
        "    joint = similarMovie_cs.merge(movie_df.toPandas(), left_on='movieId', right_on = 'movieId', how = 'inner')\n",
        "  return joint[['movieId','title','genres']]"
      ],
      "execution_count": 52,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 362
        },
        "id": "bC0pAgQQAULg",
        "outputId": "caefcec2-001f-474d-9dd7-c7ed0515d832"
      },
      "source": [
        "similarMovies(471)"
      ],
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>movieId</th>\n",
              "      <th>title</th>\n",
              "      <th>genres</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1096</td>\n",
              "      <td>Sophie's Choice (1982)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>26171</td>\n",
              "      <td>Play Time (a.k.a. Playtime) (1967)</td>\n",
              "      <td>Comedy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>55245</td>\n",
              "      <td>Good Luck Chuck (2007)</td>\n",
              "      <td>Comedy|Romance</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>160271</td>\n",
              "      <td>Central Intelligence (2016)</td>\n",
              "      <td>Action|Comedy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>536</td>\n",
              "      <td>Simple Twist of Fate, A (1994)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>4066</td>\n",
              "      <td>I'm Gonna Git You Sucka (1988)</td>\n",
              "      <td>Action|Comedy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>351</td>\n",
              "      <td>Corrina, Corrina (1994)</td>\n",
              "      <td>Comedy|Drama|Romance</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>3992</td>\n",
              "      <td>Malèna (2000)</td>\n",
              "      <td>Drama|Romance|War</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>97168</td>\n",
              "      <td>Marley (2012)</td>\n",
              "      <td>Documentary</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>515</td>\n",
              "      <td>Remains of the Day, The (1993)</td>\n",
              "      <td>Drama|Romance</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "  movieId                               title                genres\n",
              "0    1096              Sophie's Choice (1982)                 Drama\n",
              "1   26171  Play Time (a.k.a. Playtime) (1967)                Comedy\n",
              "2   55245              Good Luck Chuck (2007)        Comedy|Romance\n",
              "3  160271         Central Intelligence (2016)         Action|Comedy\n",
              "4     536      Simple Twist of Fate, A (1994)                 Drama\n",
              "5    4066      I'm Gonna Git You Sucka (1988)         Action|Comedy\n",
              "6     351             Corrina, Corrina (1994)  Comedy|Drama|Romance\n",
              "7    3992                       Malèna (2000)     Drama|Romance|War\n",
              "8   97168                       Marley (2012)           Documentary\n",
              "9     515      Remains of the Day, The (1993)         Drama|Romance"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 53
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 362
        },
        "id": "CrC_X6wff0Ck",
        "outputId": "893d8b39-04af-483a-f897-5688b5830c32"
      },
      "source": [
        "similarMovies(40491)"
      ],
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>movieId</th>\n",
              "      <th>title</th>\n",
              "      <th>genres</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>142507</td>\n",
              "      <td>Pawn Sacrifice (2015)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>72294</td>\n",
              "      <td>Christmas Carol, A (2009)</td>\n",
              "      <td>Animation|Children|Drama|Fantasy|IMAX</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>40491</td>\n",
              "      <td>Match Factory Girl, The (Tulitikkutehtaan tytt...</td>\n",
              "      <td>Comedy|Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>102800</td>\n",
              "      <td>Frances Ha (2012)</td>\n",
              "      <td>Comedy|Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>101070</td>\n",
              "      <td>Wadjda (2012)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>108188</td>\n",
              "      <td>Jack Ryan: Shadow Recruit (2014)</td>\n",
              "      <td>Action|Drama|Thriller|IMAX</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>8189</td>\n",
              "      <td>Zazie dans le métro (1960)</td>\n",
              "      <td>Comedy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>51573</td>\n",
              "      <td>Meshes of the Afternoon (1943)</td>\n",
              "      <td>Fantasy</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>166705</td>\n",
              "      <td>Fences (2016)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>114396</td>\n",
              "      <td>Cesar Chavez (2014)</td>\n",
              "      <td>Drama</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "  movieId  ...                                 genres\n",
              "0  142507  ...                                  Drama\n",
              "1   72294  ...  Animation|Children|Drama|Fantasy|IMAX\n",
              "2   40491  ...                           Comedy|Drama\n",
              "3  102800  ...                           Comedy|Drama\n",
              "4  101070  ...                                  Drama\n",
              "5  108188  ...             Action|Drama|Thriller|IMAX\n",
              "6    8189  ...                                 Comedy\n",
              "7   51573  ...                                Fantasy\n",
              "8  166705  ...                                  Drama\n",
              "9  114396  ...                                  Drama\n",
              "\n",
              "[10 rows x 3 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "usLex55pgAae"
      },
      "source": [
        "### **4. Write the report** ###"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tjlXVhfkgRuU"
      },
      "source": [
        "#### **Motivations** ####\n",
        "\n",
        "\n",
        "> The breakeout of Covid-19 revolutionally changed all aspects of lifestyles. Among them, e-commerce and info streaming have been unprecedentedly boosting. Groceories, shopping, leisures, etc online sales and services are becoming essential. The strategies of appealing customers' eyeballs, seizing business opportunies and promoting more products are heavily relying on Rececommendation System. Here, I applied Collaborative Filtering by Alternating Least Squares (ALS) model with explicit data on looking for similarities in Spark ML. It's quite a creative and interesting attempt for movie-rated users to add similarities in their bucketlist.  \n",
        "\n",
        "\n",
        "\n",
        "##### **Step 1: Data ETL and Data Exploration** #####\n",
        "> After loading data, I glimpsed every subjects to see content and data types as well as generating sums of items to roughly know the size of workload.\n",
        "##### **Step 2: OLAP** #####\n",
        "> From analytical processing, I computed multidimensional varibles to obtain some perspectives.\n",
        "##### **Step 3: Model Selection** #####\n",
        "> Having built up an ALS model, I calculated the RMSE of whole dataset without applying any hyperparameters tuning just to evaluate the model.\n",
        "##### **Step 4: Evaluate the Model** #####\n",
        "> Splitted ratings dataset between an 80% training data set and a 20% test data set. Then re-run the steps to train the model on the training set, run it on the test set, and evaluate the performance.\n",
        "##### **Step 5: Improve the Model** #####\n",
        "> After computing training set in the ALS model, the lower performance than previous one was found. But it was protected against overfitting: it will actually get this level of performance on incoming data. Tuned hyperparemers by 5-fold cross validation, applying the optimal hyperparameters on the best final model.\n",
        "##### **Step 6: Model Application: Recommend Movies** #####\n",
        "> To recommend movies for a specific user, created a function that applies the trained model, ALSModel.\n",
        "##### **Step 7: Model Application: Similary Movies Recommendation** #####\n",
        "> To find similar movies based on ALS results, compulated 2 dataframes of given movie and all movie data set by cosine similarity; ranked the result by descending and took top 10 movies.\n",
        "\n",
        "\n",
        "#### **Output and Conclusion** ####\n",
        "1. The best model for ALS has the parameters to be: maxIter=10, regParam=0.1, rank=5. The rooted mean squared error (RMSE) on the testing data is 0.88 and on the whole dataset is 0.69.\n",
        "\n",
        "2. ALS model is versatile because it not only enables to provide recommendations bu also mine latent information, which is the latent variable in matrix factorization. It's helpful in gaining some deeper insights. In this project, this information was used to measure the difference between any two movies so as to find similar movies.\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0AGHFwdqk4WW"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}