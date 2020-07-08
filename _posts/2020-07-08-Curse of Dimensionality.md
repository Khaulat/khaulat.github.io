---
Title: "What is this **Curse** in machine learning?"
Date: 2020-07-08
Tags: [machine learning, data science, feature extraction, dimension]
header:
  image: "/images/dim_reduction.png"
excerpt: "It is a concept used to describe the problems caused by high-dimensional data. What is high-dimensional data, you may ask... Read on! "
---

## What is this **Curse** in machine learning?

When we work with data in any format, there's a tendency for us to overfit or underfit when building a machine learning model. Overfitting is when the model learns too much from the data that it starts memorising it instead of understanding while underfitting is when it learns too little that it doesn't understand enough of the data. In both cases, the model is unable to generalize to other datasets.

Let us consider a simple and common machine learning dataset, the [*boston housing dataset*](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html). This data contains a total of 14 features/variables also known as the dimensions and we want to predict the prices of houses given these dimensions. We can decide to just use 2 features that are most related to the price of a house; say "the average number of rooms" and "per capita crime rate". These 2 features would give us information about the prices that  careorrect but vague, which means we haven't learnt enough from the data, thereby underfitting. As we increase the dimensions, we continue to get better accuracies but reach a point where it starts to decrease. At this point, we are leaning towards overfitting.

### Dimension = Features = Variables

The curse of dimentionality asides overfitting and underfitting is the exponential growth of space with dimensions which is another major cause of decreased accuracy with increased dimensions. As the dimention increases, the space grows exponentially leading to the need for more and more data to fill up the space. Where this data is unavailable, the space becomes sparse making the data from different dimensions far from each other. This brings up another term called *distance metrics*. 

<img src="{{ site.url }}{{ site.baseurl }}/images/data_distance.png" alt="">

Distance metrics helps algorithms (in this case, machine learning models) recognise the relationship; either similarities or differences between different data points. When these data points are far apart, as shown in the diagram above, it might perform wrongly as two similar data points might be seperated by a large space which makes them to be classified as different. 

Fortunately, there is a solution to this curse in machine learning. **Dimensionality Reduction** to the rescue! I would explain this concept and also more about distance metrics in future posts.

Please, share if you liked the post!😄 Do have a great day/night!
