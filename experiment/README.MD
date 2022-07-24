The experiments consists of a couple of phases

 * **_Crucial_ tip 1: when I run the experiments, I use `ctrl+r` to find the old command that I need, then I just slighty edit the old command where necessary (usually the subject and/or session). I almost never type out the complete commands myself.**
 * **_Crucial_ tip 2: The scripts have a protection against overwriting log files that are already there. When it already exists. It will ask whether you are sure about that. If you have to continue, the correct answer is `y` (so not `yes`). I think the best strategy is to never use session 1/2/3 for testing scripts, but use session 99 or something.**

Some more tips:
 * 
 * Marius knows where the key + locker is in which there is the money and necessary forms.
 * The forms can also be found on the "transfer computer" in the scanning room. I think it is either in `Desktop/Gilles` or `Documents/Gilles`. Karl would know.

# Session 1

## Instruction phase
 * Use the stimulus computer in the *Single Subject Room*.
 * Open Powershell
   * Press the start button and type "power". When you see powershell selected, press enter.
  * Activate the virtual environment:
    * `Documents\Gilles\virtualenv\Scripts\activate`
  * Go to the experiment folder `cd Documents\Gilles\tms_risk\experiment`
 * Run the **instruction script**: `python instruct.py --settings macbook`
 * When the subject arrives: give them first all the five forms:
   * Informed consent (only text)
   * Informed consent (form to sign)
   * Information about MRI
   * MRI checklist
   * TMS chekclist
  * After they have filled in all the forms and you checked the screening forms. 
  * Make them change into the green garments.
  * Let them read the instructions sheet. After they have done that, they can do the instruction task on the computer.

## Task phase
 .....

# Session 2/3

## Instruction phase
 * When the subject arrives: give them first all these four forms:
   * Informed consent (only text)
   * Informed consent (form to sign)
   * MRI checklist
   * TMS chekclist
  * After they have filled in all the forms and you checked the screening forms, make them change into the green garments.
  * Now they can go to the TMS preparation room.

## Task phase

### Before subject enters
 * Before the subject enters the scanning room. Prepare a bit.
 * Make sure the 
 * Open Powershell
   * Press the start button and type "power". When you see powershell selected, press enter.
 * Activate the virtual environment: `Documents\Gilles\risk_experiment\virtualenv\Scripts\activate` (*Maike*, please check this. You could also use the identical virtualenv in our stress study `C:\Users\Econ_Experimenter\Documents\SNS\Gilles\stress_risk\experiment\virtualenv\risk\Scripts\activate`)
 * Make a new trial design for the 

### When subject enters
* When Marius puts the subject in the scanner the first time, you can open the `Eye (monitor 1)` (or something like that)-script on the desktop. This will show a eye calibrations cript on the scanner display.
* Press `enter` to show the input to the camera. Marius will know what to do next.
* Once the camera is okay. 
