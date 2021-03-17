-Machine Learning based Web Application, used to predict wheather a person is likely to be diabetic or not.
-Measured on certain factors likes glucose level in body,blood pressure, age of a person,etc
-Requirements:-Streamlit (a Python library)

## <u> Let's Get Started: </u>

### Step 1. Create a Copy of this Repository
In order to work on an open-source project, you will first need to make your own copy of the repository. To do this, you should fork the repository and then clone it so that you have a local working copy.

 **Fork :fork_and_knife: this repo. Click on the Fork button at the top right corner.**

With the repository forked, you’re ready to clone it so that you have a local working copy of the code base.

 **Clone the Repository**

To make your own local copy of the repository you would like to contribute to, let’s first open up a terminal window.

We’ll use the git clone command along with the URL that points to your fork of the repository.

* Open the Command Prompt/Terminal
* Type this command:

```
git clone https://github.com/your_username/Machine-Learning-minors
```

 **Initiate the Repository**
```
git init Machine-Learning-minors
```

### Step 2: Creating a New Branch (Optional)
It is important to branch the repository so that you are able to manage the workflow, isolate your code, and control what features make it back to the main branch of the project repository.

When creating a branch, it is very important that you create your new branch off of the master branch. 
**To create a new branch, from your terminal window, follow:**

```
git branch new-branch
git checkout new-branch
```
Once you enter the git checkout command, you will receive the following output:

```
Switched to branch 'new-branch'
```

### Step 3: Contribute
Make relevant changes like fixing bug if there,enhancing the project features,etc.Contribute in any way you feel like :)

### Step 4: Commiting and Pushing:
Once you have modified an existing file or added a new file to the project, you can add it to your local repository, which we can do with the git add command.

``` git add filename``` or ``` git add .``` 

You can type the command ```git add -A``` or alternatively ```git add -all``` for all new files to be staged.


**With our file staged, we’ll want to record the changes that we made to the repository with the ```git commit``` command.**
<p> The commit message is an important aspect of your code contribution; it helps the other contributors fully understand the change you have made, why you made it, and how significant it is.  </p>
 
 ```
 git commit -m "commit message"
 ```
 
 At this point you can use the ```git push``` command to push the changes to the current branch of your forked repository:
 <br>
 a)**If working without a branch**:
 ```
 git push origin master
 ```
 b) **If working with a branch**:
 ```
 git push --set-upstream origin new-branch
 ```
 
### Step 5: Create Pull Request
At this point, you are ready to make a pull request to the original repository.

You should navigate to your **forked** repository, and press the ```“Compare & pull request”``` button on the page. 

GitHub will alert you that you are able to merge the two branches because there is no competing code. You should add in a **title**, a **comment**, and then press the **“Create pull request”** button.

### Step 6: CONGRATULATIONS :boom: :clap: :relaxed:
You have made it till the end. Kudos to you!!
