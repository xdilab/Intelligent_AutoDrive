Subject: ROAD++ progress update — VLM baseline pipeline ready, running shortly

Dear Dr. Moradi,

I wanted to share a concrete progress update since my last email. The project has moved from planning into active implementation and I now have a working baseline pipeline ready to run against the ROAD-Waymo dataset.

Since the last update, I have completed the following:

First, I finished a thorough analysis of the ROAD-Waymo annotation structure, including the full constraint sets: 49 valid duplex labels (agent+action pairs) and 86 valid triplet labels (agent+action+location combinations) in the v1.0 annotation file. These constraints are the foundation of the neuro-symbolic direction and are more numerous than the earlier version of the dataset described in the paper.

Second, I have implemented three VLM baselines using SmolVLM-500M-Instruct, a lightweight open-source vision-language model well suited to single-GPU experimentation. The first is a zero-shot run with a flat label list prompt. The second is a constraint-aware variant that injects all 135 valid labels directly into the prompt, requiring the model to select from only the semantically valid agent+action+location triplets. The third — and I think the most directly relevant to the thesis goal — feeds the verified ground-truth labels alongside the image and asks the model to produce natural language scene reasoning and an intent summary. This isolates the reasoning capability entirely from the detection problem, which means the output can be evaluated on the quality of the explanation rather than on bounding box accuracy. A dedicated conda environment is set up and all three scripts are ready to run on the two A6000 GPUs in the lab.

My immediate next step is to run the baselines and review the outputs, with a focus on the GT-conditioned reasoning results as the most direct indicator of whether this direction is worth pursuing further.

I would also like to revisit scheduling my thesis proposal defense. Please let me know what dates might work for you.

Best regards,

[Your Name]
