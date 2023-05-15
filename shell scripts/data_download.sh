#!/bin/bash -e
#SBATCH --output=%x_%j.txt --time=23:00:00 --wrap "sleep infinity"
cd /scratch/vgn2004
echo "starting download..."
curl -O https://the-eye.eu/public/AI/pile_v2/data/FreeLaw_Opinions.jsonl.zst
curl -O https://the-eye.eu/public/AI/pile_v2/data/PhilArchive.jsonl.zst
curl -O https://the-eye.eu/public/AI/pile_v2/data/NIH_ExPORTER_awarded_grant_text.jsonl.zst
curl -O https://the-eye.eu/public/AI/pile_v2/data/EuroParliamentProceedings_1996_2011.jsonl.zst
curl -O https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst
curl -O https://the-eye.eu/public/AI/pile_preliminary_components/ubuntu_irc_until_2020_9_1.jsonl.zst
curl -O https://the-eye.eu/public/AI/pile_preliminary_components/yt_subs.jsonl.zst
echo "download completed!"
