# More Explaination of Attack for Face Verification

### Threshold
Use '--threshold' to set the threshold of face verification model.  
For face-verfication model resnet101sofxmax,
it has threshold 1.244, 1.048, 0.597 for FPR 1e-3, 3e-4, 1e-4 respectively.

### Attribute
The default setting will generate adversarial examples for all attributes. 

If you only want to use one of them, just set '--adv_attribute'. It should be one of ['Blond\_Hair', 'Wavy\_Hair', 'Young', 'Eyeglasses', 'Heavy\_Makeup', 'Rosy\_Cheeks', 'Chubby', 'Mouth\_Slightly\_Open', 'Bushy\_Eyebrows', 'Wearing\_Lipstick', 'Smiling', 'Arched\_Eyebrows', 'Bangs', 'Wearing\_Earrings', 'Bags\_Under\_Eyes', 'Receding\_Hairline', 'Pale\_Skin', 'all']

### Results 
All results will be stored into the folder '--save_path'. 
 
It contains i\_k\_original\_img.png(original image), i\_k\_target\_img.png(target image) and i\_k\_True/False\_adv\_G(X,cj).png(adversarial image).  
i: original image id,  
k: target image id,  
j: attribute index

Each attack record is written into 'record.txt'.


#### All options are in the 'verification_attack.py'. 