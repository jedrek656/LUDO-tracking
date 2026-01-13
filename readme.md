# LUDO Tracker
This repository aims for solving the problem of tracking progress of LUDO's variant from a top-down video.

# Accuracy and method summary
This code works much better than I expected, and on this video it only makes one mistake (doesn't recognize dice thrown once). It doesn't read innitial position, but rather use differences on board to build it's knowledge based as a game progresses.

# Used Methods
## Board contour
Finding board is important, because this code very often filters some objects to restrict to either only ones on board, or ones that are not on the board. This section works by applying low-value hsv filter, finding contours and selecting one with the largest area. This is done once for first frame.
## Dice tracking and detection
### Crop to dice view
The dice tracking works by applying the red hsv mask, finding contours and filtering all that are inside the board. Then code finds a bounding rectangle, and crops a bit more.
### Get the digit
From a cropped dice area low saturation hsv mask is applied to get the contour for digit. This digit is compared using matchShapes, since it's tolerant to rotations.
### Compare to "ground truth"
We took one image of each side of a dice and manually cropped it. Then we found digit's contour using method mentioned above and stored it. Everytime the digit needs to be compared it refers this ground truth and chooses the smallest matchShape value.
### Decide whether new number was thrown
In order to decide if new number is thrown we assume that if ROI of dice is found it's either being thrown or already landed. We try to focus on situations when dice is visible for long time. We start with "not detected" flag. If this flag is set we look for situation where in a series of consecutive frames one digit is found at least 6 times. Every non-detected frame resets this counter. After findingh the dice flag is set to "detected". In order to reset the flag ROI has to be "not found" for at least 7 frames (setting those thresholds filter noise in form of both not finding ROI when it should, and detecting the wrong number for contour).
## Board outline searching
### Preliminary fields search
We look for circular fields using hough transform for circles on slightly gaussian-blured frame cropped to board only . This method is far from perfect. It doesn't have neither perfect precision nor recall. It's way to weak to use it for field location mapping.
### Parametric plus model
Using the fact that board is shaped as a "plus" we've created the parametric plus model (center x, center y, shorter size, longer size, rotation). In theory we could use the fact that longer size is ~2 times higher than shorter size and reduce number of variables by 1, but we decided to stick to original idea.
### Plus model fitting
We used the preliminary field searching with settings that gives high precision (more strict hough parameters), and used huber loss to reduce the impact of outliers. Then using scipy library we fitted our model using least squares.
### Getting fields
Once the parametric model is calculated, it's very easy to get fields locations. We divided longer segments for 6, and shorter for 3 points laying in equal distances. Theese are the middle-points of our found fields. For our video it showed perfect results. Then we sticked to convention where fields are numbered 0-47 with 0 being starting field of red player.
## Pawn movement search
### Detect difference
To calculate the difference between frames we used SSIM, which is method that takes much longer to execute than L2 loss, but is resistant to e.g. light distortions that makes L2 practically useless. We assumed that new position is estabilished once for 24 consecutive frames the summed value over difference map is extremely low.
### Position difference
We start with initial frame as starting position. When new position is found using method mentioned above the difference between current and old position is caluclated. Then we applied value filtering (> 0.7) and strong morpholocigal filters to be sure our shapes are filled. Then iterating over fields we check if its middlepoint is marked on this binary mask. It can return 2 points or just one. The current position becomes old position.
### Order of fields
Since code only returns two fields indexes, we needed to estabilish a way to know which way the pawn moved. For this we recalculated theese two field parameters with respect to current player's starting field location, then pawn can only move in the direction of growing number.
### Events
There are some events that can also be detected. Standing on special field is easy to detect, since they are placed in a regular places on boards, and are just returned from field index. Capture is detected when game knows that one pawn is already standing on this field. In case of only one field found there are only two possibilities: placing new pawn on board (if field is current player's home), or taking pawn to home for any other field.
## Custom video format
We implemented the custom video class which allowed us for easy composition of all tracked elements. It contains of original video with highlighted current player token (orange), vizualisation of the board, tracking the dice and custom message box for printing current events.

## Unused mechanics we considered
### Finding player bases, homes and starting fields
They were easy to find using hsv masks for certain hues, since they are very saturated. Turns out they are unnecesary to be found. 
### Cards tracking
We though of detecting playing card event by tracking cards, but now board manages it.
### Dice augmentation
We tried artificially augmenting our "ground-truth" dice images, to count for possible perspective distortion, but as we tested it did not boost the accuracy.


