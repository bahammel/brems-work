# Github commands

## Seeing changes since last save

`git stats` -- show what files have been changed  
`git diff` -- show the changes made to those files  

## Save work

`git add .` -- Stage all of the files to be saved  
`git add <file_name>` -- Stage a single file to be saved  
`git commit` -- Save file  

## Seeing past work

`git log --oneline --branches --graph --decorate` -- see a condenced list of all of the past work  
`git log --branches --graph --decorate` -- see all of the logs and the branches they are apart of  
`git log --branches --graph --decorate -p` -- see all of the logs, the branches they are apart of, and all of the changes that were made  

### Seeing a file from past work

`git show <branch_name>:<file_name> | less`  -- read through the file  
`git show <branch_name>:<file_name> > <new_file.py>`  -- Save the old file as 'new_file'  


## Resources

[https://learngitbranching.js.org/](https://learngitbranching.js.org/)  
