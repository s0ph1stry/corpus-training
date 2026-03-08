# 68K Checkpoint Samples

Step 68,000 (Run 5, v2.1 fixes active)
Generated with temperature=0.8, top_k=50, top_p=0.9

## Prompt: "The sea was"

### Sample 1
The sea wasin the _ H ang _ ; — — — and in my words , I am the old , ' I can ' t be for the day . ' " — — ' " _ No — ' s _ _ , I ' ve got no good . — " And there ' s what ' s the last . _ I _ , I ' ll tell my head . " ' I ' m so quite to be a _ H ion _ . " " _ I ' ve got to ' , " said the F ise : ' if I did not stay , and I ' ll say , and they ' ve got me . I ' ve told him . ' ' And ' I ' ll see it , ' said I , ' if you ' d

### Sample 2
The sea wasvery poor , but the night was like the night , the morning they were so much with the whole time of the morning . The great - ch ag is very much for it . I had never had a old man to say that she had not been a beautiful door in the moment , but she was a great head in the hand . It was a little way . He was about him . I had not taken up the most most happy ; but I could not do . I do very well to get on the place . I was no more and more than my eyes , I had a good thing to be ready . We should never give me a certain man , for some of the first of the moment I thought I did not know to be in a

## Prompt: "In the beginning God"

### Sample 1
In the beginning Godof the world , And be a little man of the name of God , and those who may make my heart with God . This is the best that shall be said to me , and my own father would not be , I am not a very good , but you can not tell the right of God ? For if you may , how I shall you see . But not , as you will have to be in the world . I have no means to me . But that ' s your love , my right . . . . For I think the things of the world , but that they are to be , of all the , I have not to be to make you by . But that is what we can do ? [ , and what has , as

### Sample 2
In the beginning Godas the mind , And this is not , for the knowledge of his world , which has not the nature of the same . For the same , after the body , and the death of the great , and to be all this , and the Lord of the most great . The God , and the true is and the , but the virtue of those who have been the whole ; for if you are so much all of the heart and truth , I shall say it and be that you have no evil ? And that is the God , and so he will have not been so many things , if I am not that that those who will be a good , and and it has been more in that . But that is he that if he will not be more

## Prompt: "Once upon a time there lived"

### Sample 1
Once upon a time there lived, and to see , by that of the day , so that they can have a way , but it will , for I will not not be ; I shall not say , that this was no one ? " " I do not want to speak of , " said the S care crow . " I shall have a great house of a little . " " I shall do , " said the Oh , " it is not right to be the same . " " We are so well . " " But then it is , " said the same , " you know the man , and I do not think . " " That is so good , " said the old man . " There is a boy for your good mother , " said the S care crow

### Sample 2
Once upon a time there livedwith the room of the hand , and in the sea and - place , in a way . Then the two or three minutes , a white men of five or two which did not take it for the water . Now , the other of the house , who became the old man - , and the old man of the day , that came out of the world . And now the day , the people , the voice , and the night had no longer been , " I had never found it , " he said . " I ' m not a very well , " said the time . " No , I ' ve got the mind

## Prompt: "The question of whether"

### Sample 1
The question of whetherthe whole is , or even the whole are , in the first part , and the most part of the most different , it is a good or to be in its power , but it is to be in its same of the case . The same is a great and the same , and the world that is the right and a thing of the right . The same in the thing , the one is a bad , a very particular , and to be in the world , or not not that is is to be found , and it is , it is not not in the same case . But it is not the same to say that that which has been as a very good ; for we are a little than to get a small person , but is an same

### Sample 2
The question of whetherI have , on the the other , that the the subject is not the . I know that they will be a better to make them with a right of a part of the matter . " I should have no part of this is very little , and when I were done to the same own . I saw that the other were always a old man in the state of the two and the world , and the last would be the same , and the two , in that was the first of the very most thing , and that it was in the first , I saw one of them in the same place that was that it had been in the same of the same manner in the same time , and the first and the course of the same . But the first

## Prompt: "I remember"

### Sample 1
I rememberthe old world , and I can be very simple . I do not know the world , for I have never been all the question , but I will see it , but I will have to be a good . I would , I have to be to me , for I could not see I would make it as soon as I had been , and he was a good , but for I was the good thing for any one , in the same . And then I could not understand it , and I saw a little more . I could not have a friend of the whole man , so that I should not give them the whole of my wife . I can get a little to him , and I did not know he would see it , and I should be

### Sample 2
I rememberthat I had , and he was a little one for the moment , and in this moment that I had , I had come into the night . When I had , I was going , and I was quite sure to go and go on . I was very sure , and I was right to do , and I was much one of myself , and that I had not found I was to be in the last . But I should never be a little in a great thing of a whole . But I had not found that he had been but I had been . I had a great t ic , that I was very true , to tell you I would have a poor woman ; and I was at the first time , but my father was going to be a

## Observations

**Register awareness**: Model matches register to prompt — theological for "God", fairy-tale dialogue for "Once upon a time" (generates "S care crow" and "the Oh" = Scarecrow and Oz), analytical for "The question of whether", memoir for "I remember".

**Not yet coherent**: Grammatically plausible but semantically empty. Sentences drift, referents lost. The model has learned the *music* of English prose registers but not the *content*. Expected for 13M params at 68% of Phase 1 training.

**BPE artifacts**: Named characters emerge through split tokens ("S care crow", "B at", "F ise"). The model learned co-occurrence patterns strongly enough to generate them even through BPE splitting.

**Comparison to 60K**: Qualitatively similar stage. Loss plateau at ~3.2 reflects this — difficulty curriculum is ramping (increasing corruption rate), so stable loss = the model handles harder inputs at the same quality level. Real improvement test is the 100K eval suite.

## Prompt: "The sea" (bare, 300 tokens)

### Sample 1
The sea, the B an was a great S . He was the same , and the great place was a great t al , a great b of B . _ The G . _ [ C . _ ] In the the " C . _ " This _ , " said the old old _ T m _ , " and they ' re a beautiful _ very _ , " he said . " " We must be a man in a young way ! " " But I ' m glad , " said the L ion , " so I ' m in my father . " " But I ' m the last thing , " said the first . " " A very good , " said the R at . " I am afraid , " said K . , in a good man , " that is a strong man that is to be . " " I ' m not to say I ' m true that I ' m no good to the last . " " It ' s an great good in the world , " said the B ob b ie . " You can ' t have the first way for that , " said the King . " I ' ll have to do it as I ' m afraid I ' ve thought . " " Well , what ' s the first way of the last ? " " I ' ve got to be a man . " " Yes , " said I , " if you ' d think you ' re a right , but I can ' t say . " " You must be as a

### Sample 2
The seaof the river . " I will see the little door , " said the R at . " A little voice ! " said the Oh , " I can ' t be , " said the B ob b ie . " You ' re a nice fellow . " " Oh , and if it is , " said the Alice , " you are as much as she is . " " They ' ve got to find him the place in the way of the other , " said the L ion . " I ' m a very glad to see what is . " " My boy , " said the Oh , " it is the word you ' re not . " " The same of the " B at , " said the B easy , " is all the other , the woman , but the S care crow , " I said , " and that ' s very hard to think of your things , " he said , " for he can ' t say anything . " " It ' s not an other , " said the S care crow . " And that ' s a bit , " said the P in . " He is a little , " said the " you will be a great right . He is the first thing that is very very easy . I ' m not to see a good man . " " " If you ' ve been not . " " I am a little girl , " said the " " " it ' s as a little , and I do to be an hour , and you

### Sample 3
*(Degenerate — stuck in dash repetition loop)*

### Notes on "The sea"
The bare "The sea" prompt doesn't anchor a specific register, so the model defaults to its strongest mode: fairy-tale dialogue (Oz characters). The L ion, B ob b ie, S care crow, Alice, R at, and Oh all appear — the corpus's children's literature is the most robust association. Sample 3 hit a degenerate repetition loop (dashes), which occasionally happens at temperature 0.8 when the model enters a low-entropy attractor.
