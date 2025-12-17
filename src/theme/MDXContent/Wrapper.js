/**
 * Wrapper Component for MDX Content
 *
 * This component wraps the MDX content in the Physical AI textbook with
 * additional functionality like hardware prerequisites display, personalization
 * options, and translation buttons.
 */

import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import TranslationButton from '../../components/Chapter/TranslationButton';
import PersonalizationOptions from '../../components/Chapter/PersonalizationOptions';
import HardwarePrerequisites from '../../components/Chapter/HardwarePrerequisites';
import ChatbotWidget from '../../components/Chat/ChatbotWidget';

const Wrapper = ({ children, metadata }) => {
  const { siteConfig } = useDocusaurusContext();
  const [currentLanguage, setCurrentLanguage] = React.useState('en');
  const [userHardwareProfile, setUserHardwareProfile] = React.useState(null);

  // Extract chapter metadata
  const chapterTitle = metadata?.frontMatter?.title || 'Chapter';
  const requiredHardware = metadata?.frontMatter?.requiredHardware || [];
  const recommendedHardware = metadata?.frontMatter?.recommendedHardware || [];
  const difficultyLevel = metadata?.frontMatter?.difficulty || 'intermediate';
  const contentMetadata = {
    target_hardware: requiredHardware
  };

  const handleTranslate = (language) => {
    setCurrentLanguage(language);
    // In a real implementation, this would update the content based on the language
  };

  const handleProfileChange = (profile) => {
    setUserHardwareProfile(profile);
  };

  return (
    <Layout
      title={chapterTitle}
      description={metadata?.frontMatter?.description || siteConfig.tagline}>
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--12">
            {/* Chapter Header */}
            <header className="margin-bottom--lg">
              <h1 className="text--center">{chapterTitle}</h1>

              {/* Hardware Prerequisites */}
              <HardwarePrerequisites
                chapterTitle={chapterTitle}
                requiredHardware={requiredHardware}
                recommendedHardware={recommendedHardware}
                difficultyLevel={difficultyLevel}
              />

              {/* Personalization Options */}
              <PersonalizationOptions
                userHardwareProfile={userHardwareProfile}
                onProfileChange={handleProfileChange}
                contentMetadata={contentMetadata}
              />

              {/* Translation Button */}
              <TranslationButton
                contentId={metadata?.frontMatter?.slug || 'unknown'}
                onTranslate={handleTranslate}
                currentLanguage={currentLanguage}
              />
            </header>

            {/* Main Content */}
            <main className="margin-vert--lg">
              {children}
            </main>

            {/* Chapter Footer */}
            <footer className="margin-vert--lg text--center">
              <p>
                <a href="/docs/intro">← Introduction</a> |
                <a href="/modules"> Modules</a> |
                <a href="/docs/capstone-project/project-overview"> Capstone Project →</a>
              </p>
            </footer>
          </div>
        </div>
      </div>

      {/* Chatbot Widget */}
      <ChatbotWidget initialOpen={false} />
    </Layout>
  );
};

export default Wrapper;